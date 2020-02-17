from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from pathlib import Path
from pprint import pprint
from agatha.config import config_pb2 as cpb, proto_util
from agatha.query import path_util, bow_util
from agatha.query import query_pb2 as qpb
from agatha.util import database_util
from agatha.util.sqlite3_graph import Sqlite3Graph
from agatha.util.sqlite3_bow import Sqlite3Bow


def assert_conf_has_field(config:cpb.QueryConfig, field:str)->None:
  if not config.HasField(field):
    raise ValueError(f"Must supply `{field}` term.")


if __name__ == "__main__":
  config = cpb.QueryConfig()
  proto_util.parse_args_to_config_proto(config)
  print("Running agatha query with the following custom parameters:")
  print(config)

  # Query specified
  assert_conf_has_field(config, "source")
  assert_conf_has_field(config, "target")
  #assert_conf_has_field(config, "result_path")
  print("Storing result to", config.result_path)

  result_path = Path(config.result_path)
  if result_path.is_file():
    assert config.override
  else:
    assert not result_path.exists()
    assert result_path.parent.is_dir()

  graph_index = Sqlite3Graph(config.graph_db)
  bow_index = Sqlite3Bow(config.bow_db)

  with graph_index as graph_index:
    assert config.source in graph_index, "Failed to find source in graph_index."
    assert config.target in graph_index, "Failed to find target in graph_index."


    # Get Path
    print("Finding shortest path")
    path, cached_graph = path_util.get_shortest_path(
        graph_index=graph_index,
        source=config.source,
        target=config.target,
        max_degree=config.max_degree,
    )
    if path is None:
      raise ValueError(f"Path is disconnected, {config.source}, {config.target}")
    pprint(path)

    print("Collecting Nearby Sentences")
    sentence_ids = set()
    for path_node in path:
      print("\t-", path_node)
      # Each node along the path is allowed to add some sentences
      sentence_ids.update(
        path_util.get_nearby_nodes(
          graph_index=graph_index,
          source=path_node,
          key_type=database_util.SENTENCE_TYPE,
          max_result_size=config.max_sentences_per_path_elem,
          max_degree=config.max_degree,
          cached_graph=cached_graph,
        )
      )
    sentence_ids = list(sentence_ids)

  with bow_index as bow:
    print("Downloading Sentence Text for all", len(sentence_ids), "sentences")
    # List[List[str]]
    text_corpus = [
        bow[s] for s in sentence_ids if s in bow
    ]

  print("Identifying potential query-specific stopwords")
  min_support = config.topic_model.min_support_count
  max_support = int(config.topic_model.max_support_fraction*len(text_corpus))
  term2doc_freq = bow_util.get_document_frequencies(text_corpus)
  stopwords_under = {
      t for t, c in term2doc_freq.items()
      if c < min_support
  }
  print(f"\t- {len(stopwords_under)} words occur less than {min_support} times")
  sentence_ids, text_corpus = bow_util.filter_words(
      keys=sentence_ids,
      text_corpus=text_corpus,
      stopwords=stopwords_under,
  )
  print(f"\t- Reduced to {len(text_corpus)} documents")
  assert len(sentence_ids) == len(text_corpus)

  print("Computing topics")
  word_index = Dictionary(text_corpus)
  int_corpus = [word_index.doc2bow(t) for t in text_corpus]
  topic_model = LdaMulticore(
      corpus=int_corpus,
      id2word=word_index,
      num_topics=config.topic_model.num_topics,
      random_state=config.topic_model.random_seed,
      iterations=config.topic_model.iterations,
  )

  #####################################################
  # Store results
  print("Interpreting")
  result = qpb.TopicQueryResult()
  result.source = config.source
  result.target = config.target

  # Add path
  for p in path:
    result.path.append(p)

  # Add documents from topic model
  print("\t- Topics per-document")
  for key, bow, words in zip(sentence_ids, int_corpus, text_corpus):
    doc = result.documents.add()
    doc.key = key
    for word in words:
      doc.terms.append(word)
    for topic_idx, weight in topic_model[bow]:
      topic_weight = doc.topic_weights.add()
      topic_weight.topic = topic_idx
      topic_weight.weight = weight

  # Add topics from topic model
  print("\t- Words per-topic")
  for topic_idx in range(topic_model.num_topics):
    topic = result.topics.add()
    topic.index = topic_idx
    for word_index, weight in topic_model.get_topic_terms(
        topic_idx,
        config.topic_model.truncate_size,
    ):
      term_weight = topic.term_weights.add()
      term_weight.term = topic_model.id2word[word_index]
      term_weight.weight = weight

  with open(result_path, "wb") as proto_file:
    proto_file.write(result.SerializeToString())
  print("Wrote result to", result_path)

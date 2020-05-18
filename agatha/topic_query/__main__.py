from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from pathlib import Path
from pprint import pprint
from agatha.topic_query import (
    path_util,
    bow_util,
    topic_query_result_pb2 as res_pb,
    topic_query_config_pb2 as conf_pb,
    aux_result_data,
)
from agatha.util import entity_types, proto_util
from agatha.util.sqlite3_lookup import Sqlite3Graph, Sqlite3Bow


def assert_conf_has_field(config:conf_pb.TopicQueryConfig, field:str)->None:
  if not config.HasField(field):
    raise ValueError(f"Must supply `{field}` term.")

if __name__ == "__main__":
  config = conf_pb.TopicQueryConfig()
  proto_util.parse_args_to_config_proto(config)
  print("Running agatha query with the following custom parameters:")
  print(config)

  # Query specified
  assert_conf_has_field(config, "source")
  assert_conf_has_field(config, "target")
  assert_conf_has_field(config, "result_path")

  # Double check the result path
  result_path = Path(config.result_path)
  if result_path.is_file():
    assert config.override
  else:
    assert not result_path.exists()
    assert result_path.parent.is_dir()

  # Setup the database indices
  graph_db = Sqlite3Graph(config.graph_db)
  assert config.source in graph_db, "Failed to find source in graph_db."
  assert config.target in graph_db, "Failed to find target in graph_db."

  # Preload the graph
  if config.preload_graph_db:
    print("Loading the graph in memory")
    graph_db.preload()

  # Get Path
  print("Finding shortest path")
  path, cached_graph = path_util.get_shortest_path(
      graph_index=graph_db,
      source=config.source,
      target=config.target,
      max_degree=config.max_degree,
  )
  if path is None:
    raise ValueError(f"Path is disconnected, {config.source}, {config.target}")
  pprint(path)

  ##############
  # Get text from selected sentences

  print("Collecting Nearby Sentences")
  sentence_ids = set()
  for path_node in path:
    print("\t-", path_node)
    # Each node along the path is allowed to add some sentences
    sentence_ids.update(
      path_util.get_nearby_nodes(
        graph_index=graph_db,
        source=path_node,
        key_type=entity_types.SENTENCE_TYPE,
        max_result_size=config.max_sentences_per_path_elem,
        max_degree=config.max_degree,
        cached_graph=cached_graph,
      )
    )
  sentence_ids = list(sentence_ids)

  print("Downloading Sentence Text for all", len(sentence_ids), "sentences")
  bow_db = Sqlite3Bow(config.bow_db)
  text_corpus = [
      bow_db[s] for s in sentence_ids if s in bow_db
  ]

  print("Pruning low-support words")
  min_support = config.topic_model.min_support_count
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
  dictionary = Dictionary(text_corpus)
  int_corpus = [dictionary.doc2bow(t) for t in text_corpus]
  topic_model = LdaMulticore(
      corpus=int_corpus,
      id2word=dictionary,
      num_topics=config.topic_model.num_topics,
      random_state=config.topic_model.random_seed,
      iterations=config.topic_model.iterations,
  )

  #####################################################
  # Store results

  print("Interpreting")
  result = res_pb.TopicQueryResult()
  result.source = config.source
  result.target = config.target

  # Add path
  for p in path:
    result.path.append(p)

  # Add documents from topic model
  print("\t- Topics per-document")
  for doc_id, bow, words in zip(sentence_ids, int_corpus, text_corpus):
    doc = result.documents.add()
    doc.doc_id = doc_id
    for topic_idx, weight in topic_model[bow]:
      doc.topic2weight[topic_idx] = weight

  # Add topics from topic model
  print("\t- Words per-topic")
  for topic_idx in range(topic_model.num_topics):
    topic = result.topics.add()
    for word_idx, weight in topic_model.get_topic_terms(
        topic_idx,
        config.topic_model.truncate_size,
    ):
      term = topic_model.id2word[word_idx]
      topic.term2weight[term] = weight

  print("\t- Adding Topical Network")
  aux_result_data.add_topical_network(
      result=result,
      topic_model=topic_model,
      dictionary=dictionary,
      graph_db=graph_db,
      bow_db=bow_db,
  )

  with open(result_path, "wb") as proto_file:
    proto_file.write(result.SerializeToString())
  print("Wrote result to", result_path)

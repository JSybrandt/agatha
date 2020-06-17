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
from agatha.ml.hypothesis_predictor import hypothesis_predictor
import torch
from typing import Optional

def main()->None:
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

  # Begin with result
  result = res_pb.TopicQueryResult()
  result.source = config.source
  result.target = config.target

  # Compute Agatha deep learning score
  set_metric_if_not_none(
      result=result,
      metric_name="agatha_ranking_criteria",
      value=optionally_compute_agatha_ranking_criteria(config),
  )
  compute_topic_model(config, result)

  with open(result_path, "wb") as proto_file:
    proto_file.write(result.SerializeToString())
  print("Wrote result to", result_path)

def assert_conf_has_field(config:conf_pb.TopicQueryConfig, field:str)->None:
  if not config.HasField(field):
    raise ValueError(f"Must supply `{field}` term.")

def optionally_compute_agatha_ranking_criteria(
    config:conf_pb.TopicQueryConfig
)->Optional[float]:
  print("Computing agatha_ranking_criteria")
  if not config.HasField("hypothesis_predictor"):
    print("\tSkipping agatha_ranking_criteria. Must set "
          "config.hypothesis_predictor")
    return None
  if not config.hypothesis_predictor.HasField("entity_db"):
    print("\tSkipping agatha_ranking_criteria. Must set "
          "config.hypothesis_predictor.entity_db")
    return None
  if not config.hypothesis_predictor.HasField("embedding_dir"):
    print("\tSkipping agatha_ranking_criteria. Must set "
          "config.hypothesis_predictor.embedding_dir")
    return None
  if not config.hypothesis_predictor.HasField("graph_db"):
    print("\tWarning, config.hypothesis_predictor.graph_db not set. "
          "Using config.graph_db instead. This may not be efficient.")
    config.hypothesis_predictor.graph_db = config.graph_db

  try:
    print("\tLoading:", config.hypothesis_predictor.model_path)
    model_path = Path(config.hypothesis_predictor.model_path)
    assert model_path.is_file(), f"Cannot find model_path: {model_path}"
    model = torch.load(model_path)
    assert isinstance(model, hypothesis_predictor.HypothesisPredictor), \
        "Invalid model_path. Model is not a HypothesisPredictor."
    model = model.eval()
    model.configure_paths(
        graph_db=config.hypothesis_predictor.graph_db,
        entity_db=config.hypothesis_predictor.entity_db,
        embedding_dir=config.hypothesis_predictor.embedding_dir,
        disable_cache=True
    )
    print(f"\tComputing ranking criteria for: {config.source} {config.target}")
    with torch.no_grad():
      return model.predict_from_terms([(config.source, config.target)])[0]
  except Exception as e:
    print("\tEncountered an issue when computing agatha_ranking_criteria.")
    print(f"\t{type(e).__name__}: {e}")
    print(f"Skipping agatha_ranking_criteria")
    return None

def set_metric_if_not_none(
    result:res_pb.TopicQueryResult,
    metric_name:str,
    value:Optional[float]
)->None:
  if value is not None:
    assert metric_name not in result.metrics, \
        f"Attempting to add metric: {metric_name} twice"
    result.metrics[metric_name] = value
    print("Set:", metric_name, value)


def compute_topic_model(
    config:conf_pb.TopicQueryConfig,
    result:res_pb.TopicQueryResult,
)->None:
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
  for p in path:
    result.path.append(p)
    print("\t- p")

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

if __name__ == "__main__":
  main()

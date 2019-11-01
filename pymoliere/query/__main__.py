from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore, LdaModel
from gensim.models.phrases import Phrases, Phraser
from pathlib import Path
from pprint import pprint
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.query import path_util, bow_util
from pymoliere.query import query_pb2 as qpb
from pymoliere.util import database_util
from typing import List, Set, Tuple
import itertools
import json
import sys
import pymongo


def assert_conf_has_field(config:cpb.QueryConfig, field:str)->None:
  if not config.HasField(field):
    raise ValueError(f"Must supply `{field}` term.")


if __name__ == "__main__":
  config = cpb.QueryConfig()
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere query with the following custom parameters:")
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

  print("Connecting to DB")
  database = pymongo.MongoClient(
      host=config.db.address,
      port=config.db.port,
  )[config.db.name]

  # Check that the query is in the graph.
  # Note: the graph is stored as an edge list with fields:
  # {source: "...", target: "...", weight: x}
  # This means the finds are just making sure an edge exists
  assert database.graph.find_one({"source": config.source}) is not None
  assert database.graph.find_one({"source": config.target}) is not None

  # Get Path
  print("Finding shortest path")
  path = path_util.get_shortest_path(
      collection=database.graph,
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
    # Each node along the path is allowed to add some sentences
    sentence_ids.update(
      path_util.get_nearby_nodes(
        collection=database.graph,
        source=path_node,
        key_type=database_util.SENTENCE_TYPE,
        max_result_size=config.max_sentences_per_path_elem,
        max_degree=config.max_degree,
      )
    )

  print("Downloading Sentence Text")
  text_corpus = [
      database.sentences.find_one(
        filter={"id": sent_id},
        projection={"bow":1, "_id":0}
      )["bow"]
      for sent_id in sentence_ids
  ]

  print("Identifying potential query-specific stopwords")
  min_support = config.topic_model.min_support_count
  max_support = int(config.topic_model.max_support_fraction*len(text_corpus))
  term2doc_freq = bow_util.get_document_frequencies(text_corpus)
  stopwords_under = {
      t for t, c in term2doc_freq.items()
      if c < min_support
  }
  stopwords_over = {
      t for t, c in term2doc_freq.items()
      if c > max_support
  }
  print(f"\t- {len(stopwords_under)} words occur less than {min_support} times")
  print(f"\t- {len(stopwords_over)} words occur more than {max_support} times")
  stopwords = stopwords_under.union(stopwords_over)
  sentence_ids, text_corpus = bow_util.filter_words(
      keys=sentence_ids,
      text_corpus=text_corpus,
      stopwords=stopwords,
  )
  print(f"\t- Reduced to {len(text_corpus)} documents")
  assert len(sentence_ids) == len(text_corpus)

  print("Computing topics")
  word_idx = Dictionary(text_corpus)
  int_corpus = [word_idx.doc2bow(t) for t in text_corpus]
  topic_model = LdaMulticore(
      corpus=int_corpus,
      id2word=word_idx,
      num_topics=config.topic_model.num_topics,
      random_state=config.topic_model.random_seed,
      iterations=config.topic_model.iterations,
  )

  #####################################################
  # Store results
  print("Interpreting")
  result = qpb.MoliereResult()
  result.source = config.source
  result.target = config.target

  # Add path
  for p in path:
    result.path.append(p)

  # Add documents from topic model
  print("\t- Topics per-document")
  for key, bow in zip(sentence_ids, int_corpus):
    for topic_idx, weight in topic_model[bow]:
      doc = result.documents.add()
      doc.key = key
      topic_weight = doc.topic_weights.add()
      topic_weight.topic = topic_idx
      topic_weight.weight = weight

  # Add topics from topic model
  print("\t- Words per-topic")
  for topic_idx in range(topic_model.num_topics):
    topic = result.topics.add()
    topic.index = topic_idx
    for word_idx, weight in topic_model.get_topic_terms(
        topic_idx,
        config.topic_model.truncate_size,
    ):
      term_weight = topic.term_weights.add()
      term_weight.term = topic_model.id2word[word_idx]
      term_weight.weight = weight

  with open(result_path, "wb") as proto_file:
    proto_file.write(result.SerializeToString())
  print("Wrote result to", result_path)

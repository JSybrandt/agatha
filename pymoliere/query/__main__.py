from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore, LdaModel
from gensim.models.phrases import Phrases, Phraser
from pathlib import Path
from pprint import pprint
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.query import path_util, bow_util
from pymoliere.query import query_pb2 as qpb
from pymoliere.util.db_key_util import(
    GRAPH_TYPE,
    SENTENCE_TYPE,
    key_is_type,
    to_graph_key,
    from_graph_key,
    strip_major_type,
)
from typing import List, Set, Tuple
import itertools
import json
import redis
import sys


def assert_conf_has_field(config:cpb.QueryConfig, field:str)->None:
  if not config.HasField(field):
    raise ValueError(f"Must supply `{field}` term.")


def assert_db_has_key(client:redis.Redis, key:str)->None:
  num_candidates = 5
  if not client.exists(key):
    key = strip_major_type(key)
    candidates = "\n\t".join(
      map(
        lambda s: from_graph_key(s.decode("utf-8")),
        itertools.islice(
          client.scan_iter(
            match=f"{GRAPH_TYPE}:*{key}*", # only match graph objs
          ),
          num_candidates,
        )
      )
    )
    raise ValueError(f"Failed to find {key}. Did you mean:\n\t{candidates}")


if __name__ == "__main__":
  config = cpb.QueryConfig()
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere query with the following custom parameters:")
  print(config)

  # Query specified
  assert_conf_has_field(config, "source")
  assert_conf_has_field(config, "target")
  assert_conf_has_field(config, "result_path")

  if not key_is_type(config.source, GRAPH_TYPE):
    config.source = to_graph_key(config.source)

  if not key_is_type(config.target, GRAPH_TYPE):
    config.target = to_graph_key(config.target)

  result_path = Path(config.result_path)
  if result_path.is_file():
    assert config.override
  else:
    assert not result_path.exists()
    assert result_path.parent.is_dir()

  print("Connecting to DB")
  client = redis.Redis(
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  )

  # Query is valid
  assert_db_has_key(client, config.source)
  assert_db_has_key(client, config.target)

  # Get Path
  print("Finding shortest path")
  path = path_util.get_path(
      db_client=client,
      source=config.source,
      target=config.target,
      batch_size=config.path.node_batch
  )
  if path is None:
    raise ValueError(f"Path is disconnected, {config.source}, {config.target}")
  pprint(path)

  text_keys = set()
  for path_node in path:
    text_keys.update(
      path_util.get_neighbors(
        db_client=client,
        source=path_node,
        key_type=SENTENCE_TYPE,
        max_count=config.max_sentences_per_path_elem,
        batch_size=config.path.node_batch,
      )
    )
  text_keys = list(text_keys)
  print("Retrieving text")
  with client.pipeline() as pipe:
    for key in text_keys:
      key = from_graph_key(key)
      pipe.hget(key, "bow")
    text_corpus = [json.loads(bow) for bow in pipe.execute()]
  print(f"Identified {len(text_corpus)} sentences.")

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
  text_keys, text_corpus = bow_util.filter_words(
      keys=text_keys,
      text_corpus=text_corpus,
      stopwords=stopwords,
  )
  print(f"\t- Reduced to {len(text_corpus)} documents")
  assert len(text_keys) == len(text_corpus)

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
  for key, bow in zip(text_keys, int_corpus):
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

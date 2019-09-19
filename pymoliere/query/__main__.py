from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
import redis
import sys
import itertools
from pymoliere.util.db_key_util import(
    GRAPH_TYPE,
    SENTENCE_TYPE,
    key_is_type,
    to_graph_key,
    from_graph_key,
    strip_major_type,
)
from pymoliere.query import path_util
import json
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from pprint import pprint

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

  if not key_is_type(config.source, GRAPH_TYPE):
    config.source = to_graph_key(config.source)

  if not key_is_type(config.target, GRAPH_TYPE):
    config.target = to_graph_key(config.target)

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

  print(path)

  print("Finding nearby text nodes")
  # Get Text Fields
  neighboring_text_keys = set()
  for path_node in path:
    neighboring_text_keys.update(
      path_util.get_neighbors(
        db_client=client,
        source=path_node,
        key_type=SENTENCE_TYPE,
        max_count=config.max_sentences_per_path_elem,
        batch_size=config.path.node_batch,
      )
    )
  print("Retrieving text")
  with client.pipeline() as pipe:
    for key in neighboring_text_keys:
      key = from_graph_key(key)
      pipe.hget(key, "bow")
    texts = [json.loads(bow) for bow in pipe.execute()]
  print(f"Identified {len(texts)} sentences.")

  print("Computing topics")
  # Run Topic Modeling
  print("\t- Forming Dictionary")
  word_idx = Dictionary(texts)
  print("\t- Forming Corpus")
  corpus = [word_idx.doc2bow(t) for t in texts]
  print("\t- Training Model")
  topic_model = LdaMulticore(
      corpus=corpus,
      id2word=word_idx,
      num_topics=config.topic_model.num_topics,
      random_state=config.topic_model.random_seed,
      iterations=config.topic_model.iterations,
  )
  pprint(topic_model.show_topics(formatted=False))
  # Report Topics

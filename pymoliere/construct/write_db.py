from datetime import datetime
from google.protobuf.json_format import MessageToDict
from pymoliere.config import config_pb2 as cpb
from pymoliere.construct import dask_process_global as dpg
from pymoliere.util.misc_util import Record, Edge
from pymoliere.util.misc_util import iter_to_batches
from redis import Redis
from typing import Tuple, Iterable
import json
import pymoliere


def get_redis_client_initialzizer(
    host:str,
    port:int,
    db:int,
)->Tuple[str, dpg.Initializer]:
  def _init():
    r = Redis(
      host=host,
      port=port,
      db=db,
    )
    assert r.ping()
    return r
  return "write_db:redis_client", _init


def write_edges(
    edges:Iterable[Edge],
    redis_client:Redis=None,
    batch_size:int=500,
)->Iterable[int]:
  if redis_client is None:
    redis_client = dpg.get("write_db:redis_client")
  count = 0
  with redis_client.pipeline() as pipe:
    for edge_batch in iter_to_batches(edges, batch_size):
      for edge in edge_batch:
        pipe.zadd(
          edge["source"],
          {edge["target"]: edge["weight"]},
        )
        count += 1
      pipe.execute()
  return [count]


def write_records(
    records:Iterable[Record],
    id_field:str="id",
    redis_client:Redis=None,
    batch_size:int=100,
)->Iterable[int]:
  if redis_client is None:
    redis_client = dpg.get("write_db:redis_client")
  count = 0
  with redis_client.pipeline() as pipe:
    for rec_batch in iter_to_batches(records, batch_size):
      for rec in rec_batch:
        id_ = rec[id_field]
        for field_name, value in rec.items():
          if field_name == id_field:
            continue
          if type(value) not in (int, float, str):
            value = json.dumps(value)
          pipe.hset(id_, field_name, value)
        count += 1
      pipe.execute()
  return [count]

def get_meta_record(config:cpb.ConstructConfig)->Record:
  """
  These are things we would want to store that indicate the properties of the
  system
  """
  metadata = MessageToDict(config)
  metadata["__date__"] = str(datetime.now())
  metadata["__version__"] = pymoliere.__VERSION__
  metadata["__git_head__"] = pymoliere.__GIT_HEAD__
  metadata["id"] = "__meta__"
  return metadata



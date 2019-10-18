from datetime import datetime
from google.protobuf.json_format import MessageToDict
from pymoliere.config import config_pb2 as cpb
from pymoliere.construct import dask_process_global as dpg
from pymoliere.util.misc_util import Record
from pymoliere.util.misc_util import iter_to_batches
from redis import Redis
from typing import Tuple, Iterable, Any
import json
import pymoliere
import networkx as nx


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
    subgraphs:Iterable[nx.Graph],
    redis_client:Redis=None,
)->Iterable[int]:
  """
  Writes edges to the redis database located at `write_db:redis_client`.  Edges
  are stored in a networkx graph. We iterate each node, and batch results based
  on nodes. Node weight must be specified in the "weight" attribute.
  """
  if redis_client is None:
    redis_client = dpg.get("write_db:redis_client")
  count = 0
  with redis_client.pipeline() as pipe:
    for subgraph in subgraphs:
      for source in subgraph.nodes:
        edges = {
            target: attr["weight"] for target, attr in subgraph[source].items()
        }
        pipe.zadd(source, edges)
        pipe.execute()
      count += len(subgraph.edges)
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
  metadata["id"] = "__meta__"
  return metadata


def set(key_values:Iterable[Tuple[Any, Any]])->Iterable[bool]:
  redis_client = dpg.get("write_db:redis_client")
  with redis_client.pipeline() as pipe:
    for k, v in key_values:
      pipe.set(k, v)
    return pipe.execute()


def get(keys:Iterable[Any])->Iterable[Any]:
  redis_client = dpg.get("write_db:redis_client")
  with redis_client.pipeline() as pipe:
    for k in keys:
      pipe.get(k)
    #with dpg.safe_get_worker()._lock:
    return pipe.execute()


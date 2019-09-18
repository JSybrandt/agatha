from redis import Redis
from pymoliere.util.misc_util import Record, Edge
import json
from pymoliere.construct import dask_process_global as dpg
from typing import Tuple


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

def write_edge(edge:Edge, redis_client:Redis=None)->None:
  if redis_client is None:
    redis_client = dpg.get("write_db:redis_client")
  redis_client.zadd(
    edge["source"],
    {edge["target"]: edge["weight"]},
  )


def write_record(rec:Record, id_field:str="id", redis_client:Redis=None)->None:
  if redis_client is None:
    redis_client = dpg.get("write_db:redis_client")
  id_ = rec[id_field]
  for field_name, value in rec.items():
    if field_name == id_field:
      continue
    if type(value) not in (int, float, str):
      value = json.dumps(value)
    redis_client.hset(id_, field_name, value)



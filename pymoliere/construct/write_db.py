from redis import Redis
from pymoliere.util.misc_util import Record, Edge
import json

DB_CONN = {
    "redis_client": None,
}

def init_redis_client(
    host:str,
    port:int,
    db:int,
)->None:
  DB_CONN["redis_client"] = Redis(
      host=host,
      port=port,
      db=db,
  )
  # Check that DB is around
  assert DB_CONN["redis_client"].ping()

def write_edge(edge:Edge, redis_client:Redis=None)->None:
  if redis_client is None:
    redis_client = DB_CONN["redis_client"]
  assert redis_client is not None
  redis_client.zadd(
    edge["source"],
    {edge["target"]: edge["weight"]},
  )


def write_record(rec:Record, id_field:str="id", redis_client:Redis=None)->None:
  if redis_client is None:
    redis_client = DB_CONN["redis_client"]
  assert redis_client is not None
  id_ = rec[id_field]
  for field_name, value in rec.items():
    if field_name == id_field:
      continue
    if type(value) not in (int, float, str):
      value = json.dumps(value)
    redis_client.hset(id_, field_name, value)



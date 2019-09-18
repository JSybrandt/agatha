from typing import List, Optional
from pymoliere.util.db_key_util import (
    GRAPH_TYPE,
    key_is_type,
    key_contains_type,
)
from heapdict import heapdict
from redis import Redis

def get_path(db_client:Redis, source:str, target:str)->Optional[List[str]]:
  "Gets the exact shortest path between two nodes in the network."
  assert key_is_type(source, GRAPH_TYPE)
  assert key_is_type(target, GRAPH_TYPE)
  source = source.encode()
  target = target.encode()

  pq = heapdict()
  visited = set()
  key2prev = {}  # used to recover path

  pq[source] = 0

  while len(pq) > 0:
    curr_key, curr_dist = pq.popitem()
    if curr_key == target:
      break
    visited.add(curr_key)
    for neigh_key, neigh_dist in db_client.zscan_iter(curr_key):
      if neigh_key in visited:
        continue
      new_dist = neigh_dist + curr_dist
      if neigh_key not in pq or new_dist < pq[neigh_key]:
        pq[neigh_key] = new_dist
        key2prev[neigh_key] = curr_key
  if target in key2prev:
    path = [target]
    while target in key2prev:
      target = key2prev[target]
      path.append(target)
    return [p.decode("utf-8") for p in reversed(path)]
  else: # didn't find a path
    return None


def get_neighbors(
    db_client:Redis,
    source:str,
    key_type:Optional[str]=None,
    max_count:Optional[int]=None,
)->List[str]:
  """
  Returns a collection of entity names corresponding to the nearest neighbors
  of `source`. This will extend to multi-hop neighbors.
  @param db_client: Connection to Redis server.
  @param source: Source node, must be of graph type.
  @param key_type: If supplied, only return nodes of the given type.
  @param max_count: If supplied, only return the closest X neighbors.
  @return list of graph keys, closest to furthest
  """
  assert key_is_type(source, GRAPH_TYPE)
  source = source.encode()
  pq = heapdict()
  visited = set()
  pq[source] = 0
  matches = set()
  while len(pq) > 0:
    curr_key, curr_dist = pq.popitem()
    visited.add(curr_key)
    curr_key = curr_key.decode("utf-8")
    if key_type is None or key_contains_type(curr_key, key_type):
      matches.add(curr_key)
      if max_count is not None and len(matches) >= max_count:
        break
    for neigh_key, neigh_dist in db_client.zscan_iter(curr_key):
      if neigh_key in visited:
        continue
      new_dist = neigh_dist + curr_dist
      if neigh_key not in pq or new_dist < pq[neigh_key]:
        pq[neigh_key] = new_dist
  return matches

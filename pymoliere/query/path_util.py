from typing import List, Optional, Tuple, Iterable
from pymoliere.util.db_key_util import (
    GRAPH_TYPE,
    key_is_type,
    key_contains_type,
)
from heapdict import heapdict
from redis import Redis
import networkx as nx
import heapq


def get_path(db_client:Redis, source:str, target:str, node_batch:int)->Optional[List[str]]:
  "Gets the exact shortest path between two nodes in the network."
  assert key_is_type(source, GRAPH_TYPE)
  assert key_is_type(target, GRAPH_TYPE)
  source = source.encode()
  target = target.encode()

  graph = nx.Graph()
  # dist is going to be a node-attr that indicates distance from the query terms
  graph.add_node(source, dist=0)
  graph.add_node(target, dist=0)

  def add_neighbors(root_key, neigh_key_weights):
    "Adds neighbors to root"
    for neigh_key, weight in neigh_key_weights:
      new_dist = graph.nodes[root_key]["dist"] + weight
      if neigh_key in graph:
        # If the new path is closer
        if new_dist < graph.nodes[neigh_key]["dist"]:
          graph.nodes[neigh_key]["dist"] = new_dist
      else: # add a new node
        graph.add_node(neigh_key, dist=graph.nodes[root_key]["dist"]+weight)
      graph.add_edge(root_key, neigh_key, weight=weight)

  with db_client.pipeline() as pipe:

    def get_batch(
        keys:List[str]
    )->Iterable[Tuple[str,List[Tuple[str, float]]]]:
      for k in keys:
        pipe.zscan(k)
      for k, v in zip(keys, pipe.execute()):
        yield k, v[1]

    #setup graph
    for root_key, neigh_key_weights in get_batch([source, target]):
      add_neighbors(root_key, neigh_key_weights)
    visited = {source, target}

    while not nx.has_path(graph, source, target):
      closest_unvisited_nodes = heapq.nsmallest(
          node_batch,
          filter(
            lambda x: x not in visited,
            graph.nodes,
          ),
          key=lambda k: graph.nodes[k]["dist"],
      )
      if len(closest_unvisited_nodes) == 0:
        return None
      for root_idx, neigh_key_weights in get_batch(closest_unvisited_nodes):
        add_neighbors(root_idx, neigh_key_weights)
        visited.add(root_idx)
    return nx.dijkstra_path(graph, source, target)


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

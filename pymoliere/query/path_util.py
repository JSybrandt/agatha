from typing import List, Optional, Tuple, Iterable, Set
from pymoliere.util.db_key_util import (
    GRAPH_TYPE,
    key_is_type,
    key_contains_type,
)
from heapdict import heapdict
import redis
import networkx as nx
import heapq



def get_path(db_client:redis.Redis, source:str, target:str, batch_size:int)->Optional[List[str]]:
  "Gets the exact shortest path between two nodes in the network."
  assert key_is_type(source, GRAPH_TYPE)
  assert key_is_type(target, GRAPH_TYPE)
  source = source.encode()
  target = target.encode()

  graph = nx.Graph()
  # dist is going to be a node-attr that indicates distance from the query terms
  graph.add_node(source, dist=0)
  graph.add_node(target, dist=0)

  with db_client.pipeline() as pipe:
    #setup graph
    for root_key, neigh_key_weights in _get_batch(pipe, [source, target]):
      _add_neighbors(graph, root_key, neigh_key_weights)
    visited = {source, target}

    while not nx.has_path(graph, source, target):
      close_novel_nodes = _lowest_k_dists(
          graph=graph,
          exclude=visited,
          max_res_size=batch_size,
      )
      if len(close_novel_nodes) == 0:
        return None
      for root_idx, neigh_key_weights in _get_batch(pipe, close_novel_nodes):
        _add_neighbors(graph, root_idx, neigh_key_weights)
        visited.add(root_idx)
    return [b.decode("utf-8") for b in nx.dijkstra_path(graph, source, target)]


def get_neighbors(
    db_client:redis.Redis,
    source:str,
    batch_size:int,
    max_count:int,
    key_type:Optional[str]=None,
)->List[str]:
  """
  Returns a collection of entity names corresponding to the nearest neighbors
  of `source`. This will extend to multi-hop neighbors.
  @param db_client: Connection to Redis server.
  @param source: Source node, must be of graph type.
  @param max_count: only return the closest X neighbors.
  @param key_type: If supplied, only return nodes of the given type.
  @return list of graph keys, closest to furthest
  """
  assert key_is_type(source, GRAPH_TYPE)
  source = source.encode()
  matches = set()

  def try_add_to_match(key_weights)->bool:
    for key, _ in key_weights:
      key = key.decode("utf-8")
      if key_type is None or key_contains_type(key, key_type):
        matches.add(key)
        if len(matches) == max_count:
          return True
    return False

  graph = nx.Graph()
  graph.add_node(source, dist=0)
  with db_client.pipeline() as pipe:
    # Setup
    for node, neigh_keys_weights in _get_batch(pipe, [source]):
      _add_neighbors(graph, node, neigh_keys_weights)
      if try_add_to_match(neigh_keys_weights):
        return matches
    visited = {source}

    while True:
      close_novel_nodes = _lowest_k_dists(
          graph=graph,
          exclude=visited,
          max_res_size=batch_size,
      )
      if len(close_novel_nodes) == 0:
        # explored the whole graph
        break
      for root_idx, neigh_keys_weights in _get_batch(pipe, close_novel_nodes):
        _add_neighbors(graph, root_idx, neigh_keys_weights)
        if try_add_to_match(neigh_keys_weights):
          return matches
        visited.add(root_idx)
  return matches

################################################################################

def _add_neighbors(
    graph:nx.Graph,
    root_key:str,
    neigh_key_weights:Iterable[Tuple[str, float]],
)->None:
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


def _get_batch(
    pipe:redis.client.Pipeline,
    keys:List[str],
)->Iterable[Tuple[str,List[Tuple[str, float]]]]:
  for k in keys:
    pipe.zscan(k)
  for k, v in zip(keys, pipe.execute()):
    yield k, v[1]


def _lowest_k_dists(
    graph:nx.Graph,
    exclude:Set[str],
    max_res_size:int,
)->List[str]:
  """
  Returns the nodes with the smallest attr_name value that are not present in
  exclude
  """
  return heapq.nsmallest(
      max_res_size,
      filter(
        lambda x: x not in exclude,
        graph.nodes,
      ),
      key=lambda x: graph.nodes[x]["dist"],
  )

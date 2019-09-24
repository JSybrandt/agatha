from typing import List, Optional, Tuple, Iterable, Set
from pymoliere.util.db_key_util import (
    GRAPH_TYPE,
    key_is_type,
    key_contains_type,
)
from heapdict import heapdict
import redis
import networkx as nx
import numpy as np
import heapq


def get_path(
    db_client:redis.Redis,
    source:str,
    target:str,
    batch_size:int,
)->Optional[List[str]]:
  """
  Gets the exact shortest path between two nodes in the network. This method
  runs a bidirectional search with an amortized download. At a high level, we
  are storing each node's distance to both the source and target at the same
  time. Each visit of a node leads us to identify any neighbors with shorter
  source / target distances. We know we're done when we uncover a node with a
  tightened path to both source and target.
  """
  assert key_is_type(source, GRAPH_TYPE)
  assert key_is_type(target, GRAPH_TYPE)
  source = source.encode()
  target = target.encode()

  graph = nx.Graph()
  # dist is going to be a node-attr that indicates distance from the query terms
  # downloaded indicates whether we've pulled their values from the database
  graph.add_node(
      source,
      downloaded=False,
      dists=np.array([0, np.inf]),
      visited=False
  )
  graph.add_node(
      target,
      downloaded=False,
      dists=np.array([np.inf, 0]),
      visited=False
  )
  priority = [source, target]
  # for now, we're just going to sort a list based on their graph dists

  with db_client.pipeline() as pipe:
    def get_top_undownloaded(curr_key):
      #finds the top batch that have not been downloaded yet.
      #Must download curr_key
      res = [curr_key]
      for key in priority:
        if not graph.node[key]["downloaded"]:
          res.append(key)
          if len(key) >= batch_size:
            break
      return res

    def add_batch(batch):
      # Loops through the batch and adds new nodes / edges
      # new nodes come in uninitialized
      for root_key, neigh_key_weights in _get_batch(pipe, batch):
        graph.nodes[root_key]["downloaded"] = True
        for neigh_key, neigh_weight in neigh_key_weights:
          if neigh_key not in graph:
            graph.add_node(
                neigh_key,
                #dists=graph.nodes[root_key]["dists"]+neigh_weight,
                dists=[np.inf, np.inf],
                downloaded=False,
                visited=False,
            )
          graph.add_edge(root_key, neigh_key, weight=neigh_weight)

    def get_total_dist(key):
      return np.sum(graph.nodes[key]["dists"])

    def get_min_dist(key):
      return np.min(graph.nodes[key]["dists"])

    while len(priority) > 0:
      curr_key = priority.pop(0)
      graph.nodes[curr_key]["visited"] = True
      # if we've made a connection from s to t
      if get_total_dist(curr_key) < np.inf:
        break

      # download the top x nodes in the pqueue
      if not graph.nodes[curr_key]["downloaded"]:
        add_batch(get_top_undownloaded(curr_key))
      assert graph.nodes[curr_key]["downloaded"]

      for neigh_key, edge_attr in graph[curr_key].items():
        neigh_weight = edge_attr["weight"]
        graph.nodes[neigh_key]["dists"] = np.min([
            graph.nodes[neigh_key]["dists"],
            graph.nodes[curr_key]["dists"] + neigh_weight,
          ],
          axis=1,
        )
        if not graph.nodes[neigh_key]["visited"]:
          priority.append(neigh_key)

      # update priorities
      priority = list(set(priority))
      priority.sort(key=get_min_dist)
    return [b.decode("utf-8") for b in nx.dijkstra_path(graph, source, target)]


def get_neighbors(
    db_client:redis.Redis,
    source:str,
    batch_size:int,
    max_count:int,
    key_type:Optional[str]=None,
    use_batch_ratio:float=2,
)->List[str]:
  """
  Returns a collection of entity names corresponding to the nearest neighbors
  of `source`. This will extend to multi-hop neighbors.
  @param db_client: Connection to Redis server.
  @param source: Source node, must be of graph type.
  @param max_count: only return the closest X neighbors.
  @param key_type: If supplied, only return nodes of the given type.
  @param use_batch_ratio: when the graph is small, the batch
         approximation is going to be a problem, but at some point the graph
         is large enough that we don't care. We start batching when the #
         nodes is greater than this ratio times batch size
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
          k=batch_size if len(graph.nodes) > use_batch_ratio*batch_size else 1,
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
    k:int,
)->List[str]:
  """
  Returns the nodes with the smallest attr_name value that are not present in
  exclude
  """
  return heapq.nsmallest(
      k,
      filter(
        lambda x: x not in exclude,
        graph.nodes,
      ),
      key=lambda x: graph.nodes[x]["dist"],
  )

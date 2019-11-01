from typing import List, Optional, Tuple, Iterable, Set, Callable, Any
from pymoliere.util import database_util
from heapdict import heapdict
import networkx as nx
import numpy as np
import heapq
import pymongo

def download_neighbors(
    collection:pymongo.collection.Collection,
    source:str,
    limit:int=0,
)->List[Tuple[str, float]]:
  """
  Returns the neighbors of a given node as queried from the graph. If too many
  neighbors exist, we can take a random selection based on limit. If limit is
  0, we take all.
  """
  return [
      (neigh["target"], neigh["weight"])
      for neigh
      in collection.find(
        filter={"source": source},
        projection={
          "target": 1,
          "weight": 1,
          "_id": 0,
        },
        limit=limit
      )
  ]

def recover_shortest_path_from_tight_edges(
    graph:nx.Graph,
    bridge_node:str,
)->List[str]:
  assert np.max(graph.nodes[bridge_node]["dists"]) < np.inf
  # The goal is to recover the shortest paths from the set of dists defined on
  # the graph
  def path_from_bridge(dist_idx:int)->List[str]:
    #This function traverses the graph, looking for the tight edges.
    # Start at the bridge
    path = [bridge_node]
    # While we havn't reached the source / target
    while graph.nodes[path[-1]]["dists"][dist_idx] > 0:
      # The distance of the current node to the source / target
      curr_dist = graph.nodes[path[-1]]["dists"][dist_idx]
      # for all neighbors of current node
      found_next_step = False
      for neigh, edge_attr in graph[path[-1]].items():
        neigh_dist = graph.nodes[neigh]["dists"][dist_idx]
        # The distance of the neighbor to the source / target
        # If the edge from curr_dist <- neigh_dist is equal to the distance
        # to the source.
        if np.isclose(edge_attr["weight"], curr_dist-neigh_dist):
          path.append(neigh)
          found_next_step = True
          break
      if not found_next_step:
        raise Exception("Something Terrible Happened")
    return path
  return list(reversed(path_from_bridge(0))) + path_from_bridge(1)

def get_element_with_min_criteria(
    data:Iterable[Any],
    criteria:Callable[[Any], float],
)->Any:
  min_score = np.inf
  res = None
  for element in data:
    new_score = criteria(element)
    if new_score < min_score:
      min_score = new_score
      res = element
  return res

def get_shortest_path(
    collection:pymongo.collection.Collection,
    source:str,
    target:str,
    max_degree:int,
)->Optional[List[str]]:
  """
  Gets the exact shortest path between two nodes in the network. This method
  runs a bidirectional search with an amortized download. At a high level, we
  are storing each node's distance to both the source and target at the same
  time. Each visit of a node leads us to identify any neighbors with shorter
  source / target distances. We know we're done when we uncover a node with a
  tightened path to both source and target.
  """
  graph = nx.Graph()
  # dist is going to be a node-attr that indicates distance from the query terms
  # downloaded indicates whether we've pulled their values from the database
  graph.add_node(
      # node name
      source,
      # The first is "dist to source" and the second is "dist to target"
      # a dist of np.inf indicates that we have not found a path yet.
      dists=[0, np.inf],
      # When we pop a node out of the priority queue, we'll mark it as visited
      visited=False
  )
  # Same thing, but for target node.
  graph.add_node(
      target,
      # Note how the values are swapped compared to above.
      dists=[np.inf, 0],
      visited=False
  )
  # This is our priority queue. We are going to sort this every time
  # we need a new node out of it. The criteria will be "minimim distance". As in
  # we will prioritize nodes that are close to _either_ the source or target.
  priority = set([source, target])

  while len(priority) > 0:
    curr_node = get_element_with_min_criteria(
        priority,
        lambda n: min(*graph.nodes[n]["dists"])
    )
    graph.nodes[curr_node]["visited"] = True
    priority.remove(curr_node)


    # If this node knows how to get to BOTH source and target
    if np.max(graph.nodes[curr_node]["dists"])< np.inf:
      # We're done!
      return recover_shortest_path_from_tight_edges(
          graph=graph,
          bridge_node=curr_node
      )

    # Download this node's neighbors
    # For each edge
    for neigh, weight in download_neighbors(
        collection=collection,
        source=curr_node,
        limit=max_degree,
    ):
      # We may have discovered a new node
      if neigh not in graph:
        graph.add_node(
            neigh,
            # the new node doesn't have any information. It will need to
            # be updated by a neighbor later
            dists=[np.inf, np.inf],
            visited=False
        )
      graph.add_edge(curr_node, neigh, weight=weight)

    for neigh, edge_attr in graph[curr_node].items():
      if not graph.nodes[neigh]["visited"]:
        weight = edge_attr["weight"]
        for d in range(2):
          graph.nodes[neigh]["dists"][d] = min(
            graph.nodes[neigh]["dists"][d],
            graph.nodes[curr_node]["dists"][d] + weight,
          )
        # Add new nodes to our queue
        priority.add(neigh)
  return None


def get_nearby_nodes(
    collection:pymongo.collection.Collection,
    source:str,
    max_result_size:int,
    max_degree:int,
    key_type:Optional[str]=None,
)->List[str]:
  """
  Returns a collection of entity names corresponding to the nearest neighbors
  of `source`. This will extend to multi-hop neighbors.
  @param db_client: Connection to Redis server.
  @param source: Source node, must be of graph type.
  @param max_result_size: only return the closest X neighbors.
  @param key_type: If supplied, only return nodes of the given type.
  @return list of graph keys, closest to furthest
  """
  assert len(key_type) == 1

  # We're done when this set is of the appropriate size, or the priority queue
  # is over.
  result = set()
  graph = nx.Graph()
  graph.add_node(source, dist=0, visited=False)
  priority = set([source])

  while len(priority) > 0 and len(result) < max_result_size:
    curr_node = get_element_with_min_criteria(
        priority,
        criteria=lambda n: graph.nodes[n]["dist"]
    )
    priority.remove(curr_node)
    graph.nodes[curr_node]["visited"] = True
    if key_type is None or curr_node[0] == key_type:
      result.append(curr_node)

    # For each edge
    for neigh, weight in download_neighbors(
        collection=collection,
        source=curr_node,
        limit=max_degree,
    ):
      # We may have discovered a new node
      if neigh not in graph:
        graph.add_node(
            neigh,
            # the new node doesn't have any information. It will need to
            # be updated by a neighbor later
            dist=np.inf,
            visited=False
        )
      graph.add_edge(curr_node, neigh, weight=weight)

    # At this point, we've downloaded the current node
    for neigh, edge_attr in graph[curr_node].items():
      weight = edge_attr["weight"]
      # Update the dists
      graph.nodes[neigh]["dist"] = min(
          graph.nodes[neigh]["dist"],
          graph.nodes[curr_node]["dist"] + weight,
      )
      # Add new nodes to our queue
      if not graph.nodes[neigh]["visited"]:
        priority.add(neigh)
  return result

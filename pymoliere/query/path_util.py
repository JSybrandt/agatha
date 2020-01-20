from typing import List, Optional, Tuple, Iterable, Set, Callable, Any
from pymoliere.util import database_util
from heapdict import heapdict
import networkx as nx
import numpy as np
import heapq
import pymongo
from copy import copy
from tqdm import tqdm
from pymoliere.util.sqlite3_graph import Sqlite3Graph
import random
from math import log2

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
  # Don't double count middle
  return list(reversed(path_from_bridge(0))) + path_from_bridge(1)[1:]

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
    graph_index:Sqlite3Graph,
    source:str,
    target:str,
    max_degree:int,
    disable_pbar:bool=False,
)->Tuple[Optional[List[str]], nx.Graph]:
  """
  Gets the exact shortest path between two nodes in the network. This method
  runs a bidirectional search with an amortized download. At a high level, we
  are storing each node's distance to both the source and target at the same
  time. Each visit of a node leads us to identify any neighbors with shorter
  source / target distances. We know we're done when we uncover a node with a
  tightened path to both source and target.
  """
  # pbar is just used to keep track of the # of visted nodes
  pbar = tqdm(disable=disable_pbar)
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
      visited=False,
      downloaded=False,
  )
  # Same thing, but for target node.
  graph.add_node(
      target,
      # Note how the values are swapped compared to above.
      dists=[np.inf, 0],
      visited=False,
      downloaded=False,
  )
  # This is our priority queue. We are going to sort this every time
  # we need a new node out of it. The criteria will be "minimim distance". As in
  # we will prioritize nodes that are close to _either_ the source or target.
  result = None
  priority = set([source, target])

  while len(priority) > 0:
    curr_node = get_element_with_min_criteria(
        priority,
        lambda n: min(*graph.nodes[n]["dists"])
    )
    graph.nodes[curr_node]["visited"] = True
    pbar.update(1)
    priority.remove(curr_node)


    # If this node knows how to get to BOTH source and target
    if np.max(graph.nodes[curr_node]["dists"])< np.inf:
      # We're done!
      result = recover_shortest_path_from_tight_edges(
          graph=graph,
          bridge_node=curr_node
      )
      break

    # Download this node's neighbors
    # For each edge
    #print(curr_node, graph.nodes[curr_node])
    neighborhood = graph_index[curr_node]
    if len(neighborhood) > max_degree:
      neighborhood = random.sample(neighborhood, k=max_degree)
    #print(f"Getting {len(neighborhood)} neighbors")
    for neigh in neighborhood:
      # We may have discovered a new node
      if neigh not in graph:
        graph.add_node(
            neigh,
            # the new node doesn't have any information. It will need to
            # be updated by a neighbor later
            dists=[np.inf, np.inf],
            visited=False,
            downloaded=False,
        )
      if not graph.has_edge(curr_node, neigh):
        graph.add_edge(
            curr_node,
            neigh,
            #weight=graph_index.weight(curr_node, neigh)
            weight=log2(len(neighborhood)),
        )
    graph.nodes[curr_node]["downloaded"] = True

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
  return result, graph

def clear_node_attribute(graph:nx.Graph, attribute:str, reinitialize:Any=None)->None:
  """
  Replaces the attribute with the reinitialize, or removes the attribute
  entirely if None specified
  """
  for _, attr in graph.nodes.items():
    if reinitialize is not None:
      attr[attribute] = copy(reinitialize)
    elif attribute in attr:
      del attr[attribute]

def get_nearby_nodes(
    graph_index:Sqlite3Graph,
    source:str,
    max_result_size:int,
    max_degree:int,
    key_type:Optional[str]=None,
    cached_graph:nx.Graph=None,
    disable_pbar=False,
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

  # Prepare the graph
  if cached_graph is None:
    graph = nx.Graph()
  else:
    graph = cached_graph
    clear_node_attribute(graph, "dist", np.inf)
    clear_node_attribute(graph, "visited", False)

  # This either sets or overwrites the source node
  graph.add_node(source, dist=0, visited=False)
  # The cached graph may have already downloaded this value
  if "downloaded" not in graph.nodes[source]:
    graph.nodes[source]["downloaded"] = False

  cached_count = 0
  with tqdm(total=max_result_size, disable=disable_pbar) as pbar:
    priority = set([source])
    while len(priority) > 0 and len(result) < max_result_size:
      curr_node = get_element_with_min_criteria(
          priority,
          criteria=lambda n: graph.nodes[n]["dist"]
      )
      priority.remove(curr_node)
      graph.nodes[curr_node]["visited"] = True
      if key_type is None or curr_node[0] == key_type:
        pbar.update(1)
        result.add(curr_node)
        if len(result) >= max_result_size:
          break

      # For each edge
      if not graph.nodes[curr_node]["downloaded"]:
        neighborhood = graph_index[curr_node]
        if len(neighborhood) > max_degree:
          neighborhood = random.sample(neighborhood, k=max_degree)
        #print(f"Getting {len(neighborhood)} neighbors")
        for neigh in neighborhood:
          # We may have discovered a new node
          if neigh not in graph:
            graph.add_node(
                neigh,
                # the new node doesn't have any information. It will need to
                # be updated by a neighbor later
                dist=np.inf,
                visited=False,
                downloaded=False,
            )
          if not graph.has_edge(curr_node, neigh):
            graph.add_edge(
                curr_node,
                neigh,
                #weight=graph_index.weight(curr_node, neigh)
                weight=log2(len(neighborhood)),
                #weight=1,
            )
        graph.nodes[curr_node]["downloaded"] = True
      else:
        cached_count += 1

      # At this point, we've downloaded the current node
      for neigh, edge_attr in graph[curr_node].items():
        if key_type is None or neigh[0] == key_type:
          pbar.update(1)
          result.add(neigh)
        weight = edge_attr["weight"]
        # Update the dists
        graph.nodes[neigh]["dist"] = min(
            graph.nodes[neigh]["dist"],
            graph.nodes[curr_node]["dist"] + weight,
        )
        # Add new nodes to our queue
        if not graph.nodes[neigh]["visited"]:
          priority.add(neigh)
  print("Used the cache for a bunch of neighbors:", cached_count)
  return result

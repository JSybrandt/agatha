import dask.bag as dbag
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, Set, List
from pymoliere.util import database_util, misc_util
import networkx  as nx
from collections import defaultdict
from itertools import chain
import json
import h5py
import pandas as pd
from filelock import FileLock


def get_biggraph_config(config_path:Path)->Dict[Any, Any]:
  """
  Loads the config dictionary from config_path. Pytorch BigGraph uses small
  python scripts as configs.
  """
  assert config_path.suffix == ".py", "Big graph config are executable python"
  assert config_path.is_file, "Pytorch big graph config not found."
  try:
    with open(config_path) as config_file:
      exec(config_file.read())
  except:
    raise ValueError("Could not parse and exec config file", config_path)
  assert 'get_torchbiggraph_config' in locals(), \
      "Config must define get_torchbiggraph_config"
  return locals()['get_torchbiggraph_config']()

def export_graph_for_biggraph(
    biggraph_config:Dict[Any, Any],
    graph_bag:dbag.Bag
)->Dict[Any, Any]:
  """
  Uses information supplied in the biggraph_config dictionary in order to
  inform how to best split the input graphs.
  This follows from https://arxiv.org/pdf/1903.12287.pdf and
  https://readthedocs.org/projects/torchbiggraph/downloads/pdf/latest/

  Returns a structured dict of paths
  """

  # This is the set of all variables in database_util that end in "TYPE"
  all_valid_entities = set(
      map(
        lambda name: getattr(database_util, name),
        filter(
          lambda name: name.endswith("_TYPE"),
          dir(database_util)
        )
      )
  )
  for ent_name in all_valid_entities:
    assert len(ent_name) == 1, \
        "Database util has an unexpected value ending in _TYPE"
  assert 'entities' in biggraph_config, 'Biggraph config must list entities'

  # Need to count up all the desired partitons
  total_partitions = 0
  entity2partition_count = {}
  for entity, data in biggraph_config["entities"].items():
    assert entity in all_valid_entities, \
        "Big graph config defines an entity not found in database_util"
    parts = int(data["num_partitions"])
    entity2partition_count[entity] = parts
    total_partitions += parts
  print(f"\t- Found {total_partitions} total partitions")

  assert "entity_path" in biggraph_config, \
      "Big graph config must have entity path defined"

  entity_path_dir = Path(biggraph_config["entity_path"])
  entity_path_dir.mkdir(parents=True, exist_ok=True)

  def name_to_entity(node_name:str)->str:
    assert node_name[1] == ":", "Invalid node name" + node_name
    assert node_name[0] in all_valid_entities, "Invalid node name:" + node_name
    return node_name[0]

  # We assign a node to a partition by hashing its string, and mapping that
  # number to the partition count
  def name_to_partition(node_name:str)->int:
    name_hash = misc_util.hash_str_to_int32(node_name)
    return  name_hash % entity2partition_count[name_to_entity(node_name)]

  def name_to_part_key(node_name:str)->str:
    ent = name_to_entity(node_name)
    part = name_to_partition(node_name)
    return f"{ent}_{part}"

  def nx_graphs_to_partitioned_names(
      nx_graphs:Iterable[nx.Graph]
  )->Iterable[Set[str]]:
    part_to_names = defaultdict(set)
    for graph in nx_graphs:
      for node in graph:
        part_to_names[name_to_part_key(node)].add(node)
    # return out the sets of nodes
    return list(part_to_names.values())

  def save_entity_partition(node_names:Set[str])->Dict[str, Path]:
    "Returns the partition number, because we have to return something"
    # calculate the partition given an arbitrary node
    arbitrary_node = next(iter(node_names))
    part_key = name_to_part_key(arbitrary_node)
    count_path = entity_path_dir.joinpath(f"entity_count_{part_key}.txt")
    name_path = entity_path_dir.joinpath(f"entity_names_{part_key}.json")
    with open(count_path, 'w') as count_file:
      count_file.write(str(len(node_names)))
    with open(name_path, 'w') as name_file:
      json.dump(list(node_names), name_file)
    return {"name_path": name_path, "count_path": count_path}

  print("\t- Partitioning nodes and writing entity files.")
  entity_paths = (
      graph_bag
      # Converts the networkx node into multiple sets, one per partition
      .map_partitions(nx_graphs_to_partitioned_names)
      # Merges sets by those that share a partition number
      .groupby(lambda names: name_to_part_key(next(iter(names))))
      # Unifies sets within a partition. Drops the partition number
      .map(lambda list_of_names: set(chain(*list_of_names[1])))
      # Store results
      .map(save_entity_partition)
      .compute()
  )

  # At this point, there is a type / part file for everything
  # Now, we need to handle edges
  assert "relations" in biggraph_config, \
      "BigGraph config must have relations defined"
  for relation in biggraph_config["relations"]:
    assert relation["lhs"] in all_valid_entities, \
        "Config has a relationship defined on a non-existing entity."
    assert relation["rhs"] in all_valid_entities, \
        "Config has a relationship defined on a non-existing entity."
  # Relations are stored as two-character strings
  relation2idx = {
      f"{relation_data['lhs']}{relation_data['rhs']}": relation_idx
      for relation_idx, relation_data in enumerate(biggraph_config["relations"])
  }

  def load_names_to_idx(part_key:str)->Dict[str, int]:
    name_path = entity_path_dir.joinpath(f"entity_names_{part_key}.json")
    assert name_path.is_file()
    with open(name_path, 'r') as name_file:
      return {name: idx for idx, name in enumerate(json.load(name_file))}

  def get_relation_key(lhs:str, rhs:str)->str:
    return name_to_entity(lhs) + name_to_entity(rhs)

  def get_bucket_key(lhs:str, rhs:str)->str:
    return f"{name_to_partition(lhs)}_{name_to_partition(rhs)}"

  def is_part_key_related_to_bucket_key(part_key:str, bucket_key:str)->bool:
    _, part_idx = part_key.split("_")
    lhs_part, rhs_part = bucket_key.split("_")
    return part_idx in {lhs_part, rhs_part}

  def nx_graphs_to_edges(
      graphs:Iterable[nx.Graph]
  )->Iterable[Tuple[str, List[Tuple[int, int, int]]]]:
    """
    Iterates the graphs and collects each edge based on its bucket.
    Then, it loads the respective entity information necessary to deduce the
    correct local indices for each node.
    Finally, it writes out the partial edge buckets, ready to be grouped-by
    """
    unique_part_keys = set()
    bucket_key2edges = defaultdict(list)
    for graph in graphs:
      for a, b in graph.edges:
        unique_part_keys.add(name_to_part_key(a))
        unique_part_keys.add(name_to_part_key(b))
        comps = [(a, b)]
        if name_to_entity(a) != name_to_entity(b):
          comps.append((b, a))
        for lhs, rhs in comps:
          relation_key = get_relation_key(lhs, rhs)
          if relation_key in relation2idx:
            bucket_key = get_bucket_key(lhs, rhs)
            ridx = relation2idx[relation_key]
            bucket_key2edges[bucket_key].append([ridx, lhs, rhs])
    # Get the set of partitions associated with this data
    for part_key in unique_part_keys:
      def should_replace(name):
        return type(name) == str and name_to_part_key(name) == part_key
      name2idx = load_names_to_idx(part_key)
      for edges in bucket_key2edges.values():
        for edge in edges:
          # lhs
          if should_replace(edge[1]):
            edge[1] = name2idx[edge[1]]
          # rhs
          if should_replace(edge[2]):
            edge[2] = name2idx[edge[2]]
    return list(bucket_key2edges.items())

  assert "edge_paths" in biggraph_config, "Config must supply edge paths"
  assert len(biggraph_config["edge_paths"]) == 1, \
      "We only support a single edge path currently"
  edge_dir = Path(biggraph_config["edge_paths"][0])
  edge_dir.mkdir(exist_ok=True, parents=True)

  tmp_edge_dir = edge_dir.joinpath("tmp_files")
  tmp_edge_dir.mkdir(exist_ok=True, parents=True)


  def write_to_tmp_edge_files(bucket_edges)->Path:
    bucket_key, edge_list = bucket_edges
    tmp_edge_path = tmp_edge_dir.joinpath(f"{bucket_key}")
    lock_path = tmp_edge_dir.joinpath(f".{bucket_key}.lock")
    with FileLock(lock_path):
      with open(tmp_edge_path, 'a') as edge_file:
        for rel, lhs, rhs in edge_list:
          edge_file.write(f"{rel}\t{lhs}\t{rhs}\n")
    return tmp_edge_path

  def tmp_file_to_bucket(tmp_edges_path:Path)->Path:
    """
    Store the resulting edge list h5 file!
    """
    bucket_key = tmp_edges_path.name
    edge_path = edge_dir.joinpath(f"edges_{bucket_key}.h5")
    rel_ds = []
    lhs_ds = []
    rhs_ds = []
    with open(tmp_edges_path) as tmp_file:
      for line in tmp_file:
        rel, lhs, rhs = list(map(int, line.strip().split("\t")))
        rel_ds.append(rel)
        lhs_ds.append(lhs)
        rhs_ds.append(rhs)
    with h5py.File(edge_path, 'w') as edge_file:
      edge_file.create_dataset("rel", data=rel_ds)
      edge_file.create_dataset("lhs", data=lhs_ds)
      edge_file.create_dataset("rhs", data=rhs_ds)
      edge_file.attrs["format_version"] = 1
    return str(edge_path)

  print("\t- Bucketing edges, converting to local indices.")
  bucket_paths = (
      graph_bag
      # Index and bucket edges
      .map_partitions(nx_graphs_to_edges)
      # performs groupby using storage
      .map(write_to_tmp_edge_files)
      .distinct()
      .map(tmp_file_to_bucket)
      .compute()
  )
  return {"entity_paths": entity_paths, "bucket_paths": bucket_paths}

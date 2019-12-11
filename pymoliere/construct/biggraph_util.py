import dask
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
from sqlitedict import SqliteDict
from functools import lru_cache


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
  ####################################33
  # Helper data

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
  entity2partition_count = {}
  for entity, data in biggraph_config["entities"].items():
    assert entity in all_valid_entities, \
        "Big graph config defines an entity not found in database_util"
    parts = int(data["num_partitions"])
    entity2partition_count[entity] = parts

  assert "entity_path" in biggraph_config, \
      "Big graph config must have entity path defined"

  entity_dir = Path(biggraph_config["entity_path"])
  entity_dir.mkdir(parents=True, exist_ok=True)

  entity_db_dir = entity_dir.joinpath("tmp_files")
  entity_db_dir.mkdir(parents=True, exist_ok=True)

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

  assert "edge_paths" in biggraph_config, "Config must supply edge paths"
  assert len(biggraph_config["edge_paths"]) == 1, \
      "We only support a single edge path currently"
  edge_dir = Path(biggraph_config["edge_paths"][0])
  edge_dir.mkdir(exist_ok=True, parents=True)

  tmp_edge_dir = edge_dir.joinpath("tmp_files")
  tmp_edge_dir.mkdir(exist_ok=True, parents=True)

  ##################################################
  # Helper functions

  def name_to_entity(node_name:str)->str:
    assert node_name[1] == ":"
    assert node_name[0] in all_valid_entities
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

  def get_relation_key(lhs:str, rhs:str)->str:
    return name_to_entity(lhs) + name_to_entity(rhs)

  def get_bucket_key(lhs:str, rhs:str)->str:
    return f"{name_to_partition(lhs)}_{name_to_partition(rhs)}"

  def is_part_key_related_to_bucket_key(part_key:str, bucket_key:str)->bool:
    _, part_idx = part_key.split("_")
    lhs_part, rhs_part = bucket_key.split("_")
    return part_idx in {lhs_part, rhs_part}

  ##################################################
  # Step 1, partition nodes

  def nx_graph_to_partitioned_node_names(nx_graphs:Iterable[nx.Graph]):
    part_key2names = defaultdict(set)
    # All of my local nodes
    for graph in nx_graphs:
      for node in graph:
        part_key2names[name_to_part_key(node)].add(node)
    return list(part_key2names.values())

  def node_set_to_part_key(node_names:Set[str])->str:
    return name_to_part_key(next(iter(node_names)))

  partitioned_node_names = (
      graph_bag
      .map_partitions(nx_graph_to_partitioned_node_names)
      .foldby(node_set_to_part_key, set.union)
  )
  partitioned_node_names = partitioned_node_names.persist()

  ##################################################
  # Step 2, create count and json files

  def write_json_and_count_files(partitioned_node_names)->Dict[str, int]:
    # Get part_key from name
    node_name_to_local_idx = {}
    for part_key, node_names in partitioned_node_names:
      node_names = list(node_names)
      for local_idx, node_name in enumerate(node_names):
        node_name_to_local_idx[node_name] = local_idx
      count_path = entity_dir.joinpath(f"entity_count_{part_key}.txt")
      name_path = entity_dir.joinpath(f"entity_names_{part_key}.json")
      with open(count_path, 'w') as count_file:
        count_file.write(str(len(node_names)))
      with open(name_path, 'w') as name_file:
        json.dump(list(node_names), name_file)
    return node_name_to_local_idx

  node2local_idx = dask.delayed(write_json_and_count_files)(partitioned_node_names)

  ##################################################
  # Step 3, bucket edges

  def nx_graph_to_bucketed_edges(
      graphs:Iterable[nx.Graph],
      node2local:Dict[str, int],
  )->Iterable[int]:
    bucket_key2edges = defaultdict(list)
    for graph in graphs:
      for name_a, name_b in graph.edges:
        comps = [(name_a, name_b)]
        if name_to_entity(name_a) != name_to_entity(name_b):
          comps.append((name_b, name_a))
        # For both x->y and y->x comparisons
        for lhs, rhs in comps:
          relation_key = get_relation_key(lhs, rhs)
          if relation_key in relation2idx:
            bucket_key = get_bucket_key(lhs, rhs)
            ridx = relation2idx[relation_key]
            bucket_key2edges[bucket_key].append([
              relation2idx[relation_key],
              node2local[lhs],
              node2local[rhs],
            ])
    total_written_edges = 0
    for bucket_key, edge_list in bucket_key2edges.items():
      tmp_edge_path = tmp_edge_dir.joinpath(f"{bucket_key}.txt")
      lock_path = tmp_edge_dir.joinpath(f".{bucket_key}.lock")
      with FileLock(lock_path):
        with open(tmp_edge_path, 'a') as edge_file:
          for rel, lhs, rhs in edge_list:
            edge_file.write(f"{rel}\t{lhs}\t{rhs}\n")
            total_written_edges += 1
    return [total_written_edges]

  print("\t- Bucketing edges")
  total_written_edges = sum(
      graph_bag
      # Index and bucket edges
      .map_partitions(nx_graph_to_bucketed_edges, node2local_idx)
      .compute()
  )
  del node2local_idx


  def tmp_file_to_bucket(tmp_edges_path:Path)->int:
    """
    Store the resulting edge list h5 file!
    """
    # Get bucket key from "{bucket_key}.txt"
    bucket_key = tmp_edges_path.name.split(".")[0]
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
    return 1

  print("\t- Writing h5")
  total_buckets = sum(
      dbag.from_sequence(list(tmp_edge_dir.glob("*.txt")))
      .map(tmp_file_to_bucket)
      .compute()
  )
  print(f"\t\t- Wrote {total_buckets} buckets")

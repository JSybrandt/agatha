import dask
import dask.bag as dbag
from pathlib import Path
from typing import Dict, Any, Tuple, Iterable, Set, List
from agatha.util import database_util, misc_util
import networkx  as nx
from collections import defaultdict
from itertools import chain, permutations, product
import json
import h5py
import pandas as pd
from filelock import FileLock
from sqlitedict import SqliteDict
from functools import lru_cache
from copy import copy
from agatha.util.misc_util import iter_to_batches
import pickle
import random
import string

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

def get_valid_entity_symbols()->Set[str]:
  symbols = set(map(
    lambda name: getattr(database_util, name),
    filter(lambda name: name.endswith("_TYPE"), dir(database_util))
  ))
  # Check to make sure nothing weird happened in the database_util
  for ent_name in symbols:
    assert len(ent_name) == 1, \
        "Database util has an unexpected value ending in _TYPE"
  return symbols

def get_partition_count(biggraph_config:Dict[Any, Any])->int:
  valid_symbols = get_valid_entity_symbols()
  partition_count = None
  for entity, data in biggraph_config["entities"].items():
    assert entity in valid_symbols, "Config contains invalid symbol"
    this_part_count = int(data["num_partitions"])
    if partition_count is None:
      partition_count = this_part_count
    else:
      assert partition_count == this_part_count, \
          "Due to technical limitations, all ents must have same num_parts"
  assert partition_count is not None, "Must have at least one entity"
  return partition_count

def get_used_entity_symbols(biggraph_config:Dict[Any, Any])->Set[str]:
  valid_symbols = get_valid_entity_symbols()
  res = set()
  for entity in biggraph_config["entities"]:
    assert entity in valid_symbols, "Config contains invalid symbol"
    res.add(entity)
  return res

def index_relationships(biggraph_config:Dict[Any, Any])->Dict[str, int]:
  valid_symbols = get_valid_entity_symbols()
  for relation in biggraph_config["relations"]:
    assert relation["lhs"] in valid_symbols, \
        "Config has a relationship defined on a non-existing entity."
    assert relation["rhs"] in valid_symbols, \
        "Config has a relationship defined on a non-existing entity."
  return {
      f"{rel['lhs']}{rel['rhs']}": idx
      for idx, rel in enumerate(biggraph_config["relations"])
  }

def export_graph_for_biggraph(
    biggraph_config:Dict[Any, Any],
    graph_bag:dbag.Bag
)->Dict[Any, Any]:
  # Basic checks
  assert 'entities' in biggraph_config, \
      'Biggraph config must list entities'
  assert "entity_path" in biggraph_config, \
      "Big graph config must have entity path defined"
  assert "relations" in biggraph_config, \
      "BigGraph config must have relations defined"
  assert "edge_paths" in biggraph_config, "Config must supply edge paths"
  assert len(biggraph_config["edge_paths"]) == 1, \
      "We only support a single edge path currently"

  # Helper info that lets us coordinate
  partition_count = get_partition_count(biggraph_config)
  relation2idx = index_relationships(biggraph_config)
  entity_symbols = get_used_entity_symbols(biggraph_config)

  # This is where we store the entity files (counts and json lists)
  entity_dir = Path(biggraph_config["entity_path"])
  # This is where we store the edge h5 buckets
  edge_dir = Path(biggraph_config["edge_paths"][0])
  # This is were we store the intermediate inverted index
  name2local_path = entity_dir.joinpath("name2local.pkl")
  # We're going to need some space to coordinate edges
  edge_tmp_dir = edge_dir.joinpath("tmp")
  # We need to record where we put the tmp edge files
  tmp_edges_done_path = edge_tmp_dir.joinpath("__done__.pkl")

  # Make sure those dirs exist
  entity_dir.mkdir(parents=True, exist_ok=True)
  edge_dir.mkdir(exist_ok=True, parents=True)
  edge_tmp_dir.mkdir(exist_ok=True, parents=True)

  # For each bucket, we want to have a dir
  bk_key2tmp_edge_dir = {}
  for l_part, r_part in product(range(partition_count), range(partition_count)):
    bucket_key = f"{l_part}_{r_part}"
    tmp_bucket_dir = edge_tmp_dir.joinpath(bucket_key)
    tmp_bucket_dir.mkdir(parents=True, exist_ok=True)
    bk_key2tmp_edge_dir[bucket_key] = tmp_bucket_dir

  # Helper functions

  def save(data, path):
    with open(path, 'wb') as f:
      pickle.dump(data, f)

  def load(path):
    with open(path, 'rb') as f:
      return pickle.load(f)

  def node_to_entity(node_name:str)->str:
    assert node_name[1] == ":"
    assert node_name[0] in entity_symbols
    return node_name[0]

  def node_to_partition(node_name:str)->int:
    return misc_util.hash_str_to_int32(node_name) % partition_count

  def node_to_part_key(node_name:str)->str:
    return f"{node_to_entity(node_name)}_{node_to_partition(node_name)}"

  def edge_to_relation_key(edge)->str:
    lhs, rhs = edge
    return node_to_entity(lhs) + node_to_entity(rhs)

  def edge_to_bucket_key(edge):
    lhs, rhs = edge
    return f"{node_to_partition(lhs)}_{node_to_partition(rhs)}"

  def graphs_to_nodes(graphs):
    res = set()
    for graph in graphs:
      for node in graph:
        res.add(node)
    return res

  def write_json_and_count_files(fold_by_result):
    part_key, nodes = fold_by_result
    nodes = sorted(nodes)
    count_path = entity_dir.joinpath(f"entity_count_{part_key}.txt")
    name_path = entity_dir.joinpath(f"entity_names_{part_key}.json")
    with open(count_path, 'w') as count_file:
      count_file.write(str(len(nodes)))
    with open(name_path, 'w') as name_file:
      json.dump(nodes, name_file)
    return name_path

  def get_name2local():
    # Loads the inverse index, and keeps it around
    if hasattr(get_name2local, 'name2local'):
      name2local = get_name2local.name2local
    else:
      with open(name2local_path, 'rb') as pickle_file:
        name2local = get_name2local.name2local = pickle.load(pickle_file)
    return name2local

  def write_name2local(total_fold_by_results):
    name2local = {}
    for _, nodes in total_fold_by_results:
      for idx, node in enumerate(sorted(nodes)):
        name2local[node] = idx
    save(name2local, name2local_path)

  def graphs_to_final_edges(graphs):
    # Loads each graph and looks up the int info eventually written to h5
    name2local = get_name2local()
    res = []
    for graph in graphs:
      for a, b in graph.edges():
        # Iterating the edges in the graph only gives us one side
        for edge in [(a, b), (b, a)]:
          rel_key = edge_to_relation_key(edge)
          if rel_key in relation2idx:
            try:
              bucket_key = edge_to_bucket_key(edge)
              rel = relation2idx[rel_key]
              lhs = name2local[edge[0]]
              rhs = name2local[edge[1]]
              res.append((bucket_key, rel, lhs, rhs))
            except:
              print("ERROR WITH", edge)
    return res

  def write_tmp_edge_file(fold_by_result):
    bucket_key, edges = fold_by_result
    # select a file name that is 10 random characters
    tmp_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
    tmp_edge_path = bk_key2tmp_edge_dir[bucket_key].joinpath(f"{tmp_name}.pkl")
    save(edges, tmp_edge_path)
    return tmp_edge_path

  def tmp_edges_to_h5(bucket_key, partial_paths):
    rel = []
    lhs = []
    rhs = []
    for path in partial_paths:
      for _, rel_idx, lhs_idx, rhs_idx in load(path):
        rel.append(rel_idx)
        lhs.append(lhs_idx)
        rhs.append(rhs_idx)
    # write
    edge_path = edge_dir.joinpath(f"edges_{bucket_key}.h5")
    with h5py.File(edge_path, 'w') as edge_file:
      edge_file.create_dataset("rel", data=rel)
      edge_file.create_dataset("lhs", data=lhs)
      edge_file.create_dataset("rhs", data=rhs)
      edge_file.attrs["format_version"] = 1
    return edge_path

  def merge_sets_binop(total, new):
    if type(new) is set:
      total = total.union(new)
    else:
      total.add(new)
    return total

  def inverted_idx_from_fold_result(fold_res, query):
    for key, data in fold_res:
      if key == query:
        return {name: idx for idx, name in enumerate(data)}
    raise Exception("Invalid Key!")

  # Time to compute!

  # All at once, we need to group the nodes by part key

  if not name2local_path.is_file():
    print(f"\t\t- Partitioning names")
    names_grouped_by_part = (
        graph_bag
        .map_partitions(graphs_to_nodes)
        .foldby(
          key=node_to_part_key,
          initial=set,
          binop=merge_sets_binop,
          combine=set.union,
          split_every=256,
        )
    )
    dask.compute([
      names_grouped_by_part.map_partitions(write_name2local),
      names_grouped_by_part.map(write_json_and_count_files)
    ])


  if not tmp_edges_done_path.is_file():
    print(f"\t\t- Writing temp edge files")
    tmp_edge_files = (
      graph_bag
      .map_partitions(graphs_to_final_edges)
      # group by produces a record per key per part
      .groupby(lambda edge: edge[0])
      .map(write_tmp_edge_file)
      .compute()
    )
    save(tmp_edge_files, tmp_edges_done_path)
  else:
    tmp_edge_files = load(tmp_edges_done_path)

  bucket2tmp_paths = defaultdict(list)
  for path in tmp_edge_files:
    bucket2tmp_paths[path.parent.name].append(path)

  print("\t\t- Writing final h5 files")
  dask.compute([
    dask.delayed(tmp_edges_to_h5)(bucket_key, paths)
    for bucket_key, paths
    in bucket2tmp_paths.items()
  ])

#!/usr/bin/env python3
"""
Extracts Graph JSON files from an Sqlite3 File.

This is required to perform large-scale graph operations on a graph that has
already been collected into an sqlite3 database.  Typically, we discard the
intermediate graph files that are used to create these large graph databases.

"""

from pathlib import Path
from agatha.util.sqlite3_lookup import Sqlite3Graph
from agatha.util.misc_util import iter_to_batches
from fire import Fire
import json
from tqdm import tqdm


def main(
    input_db:Path,
    output_dir:Path,
    nodes_per_file:int=1e6,
    output_file_fmt_str:str="{:08d}.txt",
    disable_pbar:bool=False
):
  """Sqlite3Graph -> Edge Json

  Args:
    input_db: A graph sqlite3 table.
    output_dir: The location of a directory that we are going to make and fill
      with json files.
    nodes_per_file: Each file is going to contain at most this number of nodes.
    output_file_fmt_str: This string will be called with `.format(int)` for
      each output file. Must produce unique names for each string.
  """

  input_db = Path(input_db)
  assert input_db.is_file(), f"Failed to find {input_db}"

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  assert output_dir.is_dir(), f"Failed to find dir: {output_dir}"
  assert len(list(output_dir.iterdir())) == 0, f"{output_dir} is not empty."

  nodes_per_file = int(nodes_per_file)
  assert nodes_per_file > 0, \
      "Must supply positive number of edges per output file."

  try:
    format_test = output_file_fmt_str.format(1)
  except Exception:
    assert False, "output_file_fmt_str must contain format component `{}`"

  print("Opening", input_db)
  graph = Sqlite3Graph(input_db)
  if not disable_pbar:
    num_nodes = len(graph)
    graph = tqdm(graph, total=num_nodes)
    graph.set_description("Reading edges")

  for batch_idx, edge_batch in enumerate(
      iter_to_batches(graph, nodes_per_file)
  ):
    file_path = output_dir.joinpath(output_file_fmt_str.format(batch_idx+1))
    if disable_pbar:
      print(file_path)
    else:
      graph.set_description(f"Writing {file_path}")
    assert not file_path.exists(), \
      f"Error: {file_path} already exists. Is your format string bad?"
    with open(file_path, 'w') as json_file:
      for node, neighbors in edge_batch:
        for neigh in neighbors:
          json_file.write(json.dumps({"key": node, "value": neigh}))
          json_file.write("\n")
    if not disable_pbar:
      graph.set_description("Reading edges")
  print("Done!")

if __name__ == "__main__":
  Fire(main)

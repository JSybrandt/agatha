#!/usr/bin/env python3

from fire import Fire
from agatha.util import sqlite3_lookup
from pathlib import Path
import dask.bag as dbag
from typing import Iterable, Dict, Tuple, Any, Optional
import json
import shutil

def path_to_type_part(path:Path)->Tuple[str, int]:
  entity, names, typ, part = path.stem.split("_")
  assert entity == "entity"
  assert names == "names"
  assert len(typ) == 1
  part = int(part)
  return typ, part

def parse_path(path:Path)->Iterable[Dict[str, Dict[str, Any]]]:
  path = Path(path)
  assert path.is_file()
  assert path.suffix == ".json"
  with open(path, 'r') as json_path:
    entities = json.load(json_path)
  typ, part = path_to_type_part(path)
  for row, ent in enumerate(entities):
    yield dict(
        key=ent,
        value=dict(
          type=typ,
          part=part,
          row=row
        )
    )

def paths_to_kvs(paths:Iterable[Path])->Iterable[Dict[str, Dict[str,str]]]:
  res = []
  for path in paths:
    for kv in parse_path(path):
      res.append(kv)
  return res

def convert(
    entity_dir:Path,
    output_db_path:Path,
    subset_types:Optional[str]=None,
    agatha_install_path:Path=Path("../../"),
)->None:
  """
  Creates a database that maps name -> type, part, row

  Args:
    entity_dir: Directory containing files named
      `entity_names_<type>_<part>.json`.
    output_db_path: Location to write entity sqlite3 lookup table.
    subset_types: String containing type characters to select. For instance,
      `--subset_types mp` will only index the embeddings needed to train
      the agatha deep learning model.
    agatha_install_path: The place you cloned the agatha repo. Should contain
      the `tools` subdir. Make sure you run `make` before running this tool.
  """

  entity_dir = Path(entity_dir)
  output_db_path = Path(output_db_path)
  assert entity_dir.is_dir(), "Failed to find entity dir"
  assert not output_db_path.exists(), "Cannot overwrite db path"

  intermediate_data_dir = output_db_path.parent.joinpath("tmp")
  intermediate_data_dir.mkdir(parents=True, exist_ok=True)

  name_json_paths = list(entity_dir.glob("entity_names_*.json"))
  assert any(name_json_paths), "Failed to find entity_names_*.json"
  if subset_types is not None:
    assert isinstance(subset_types, str)
    subset_types = set(subset_types)
    init_size = len(name_json_paths)
    name_json_paths = [
        p for p in name_json_paths
        if path_to_type_part(p)[0] in subset_types
    ]
    final_size = len(name_json_paths)
    print(f"Subset {init_size} entity files down to {final_size}")

  entity_location_kvs = (
      dbag.from_sequence(name_json_paths)
      .map_partitions(paths_to_kvs)
  )
  sqlite3_lookup.create_lookup_table(
      key_value_records=entity_location_kvs,
      result_database_path=output_db_path,
      intermediate_data_dir=intermediate_data_dir,
      agatha_install_path=agatha_install_path,
      verbose=True,
  )
  shutil.rmtree(intermediate_data_dir)


if __name__ == "__main__":
  Fire(convert)

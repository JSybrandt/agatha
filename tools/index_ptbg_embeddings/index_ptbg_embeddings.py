#!/usr/bin/env python3

from fire import Fire
from agatha.util import sqlite3_lookup
from pathlib import Path
import dask.bag as dbag
from typing import Iterable, Dict, Tuple, Any
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
    agatha_install_path:Path=Path("../../"),
)->None:
  """
  Creates a database that maps name -> type, part, row
  """

  entity_dir = Path(entity_dir)
  output_db_path = Path(output_db_path)
  assert entity_dir.is_dir(), "Failed to find entity dir"
  assert not output_db_path.exists(), "Cannot overwrite db path"

  intermediate_data_dir = output_db_path.parent.joinpath("tmp")
  intermediate_data_dir.mkdir(parents=True, exist_ok=True)

  name_json_paths = list(entity_dir.glob("entity_names_*.json"))
  assert any(name_json_paths), "Failed to find entity_names_*.json"
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

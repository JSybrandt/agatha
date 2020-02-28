import sqlite3
from pathlib import Path
import json
from typing import List, Any
import dask.bag as dbag
from agatha.util.misc_util import Record
import os

def _record_to_kv_json(
    record:Record,
    key_field:str,
    value_field:str
)->str:
  return json.dumps((
      str(record[key_field]),
      str(record[value_field]),
    ))

def _make_sqlite3_database_from_json(
    intermediate_data_dir:Path,
    database_path:Path,
    agatha_install_path:Path,
)->None:
  create_lookup_table_exec = (
      agatha_install_path
      .joinpath("tools")
      .joinpath("create_lookup_table")
      .joinpath("create_lookup_table")
  )
  assert create_lookup_table_exec.is_file(), \
      "Failed to find create_lookup_table tool. Incorrect install path?"
  assert intermediate_data_dir.is_dir()
  assert not database_path.exists()
  flag = os.system(
      f"{create_lookup_table_exec} "
      f"-i {intermediate_data_dir} "
      f"-o {database_path} "
  )
  assert flag == 0, "Something failed during create_lookup_table"
  assert database_path.is_file(), "Failed to create database"

def create_lookup_table(
  record_bag: dbag.Bag,
  key_field:str,
  value_field:str,
  database_path:Path,
  intermediate_data_dir:Path,
  agatha_install_path:Path,
)->None:
  database_path = Path(database_path)
  intermediate_data_dir = Path(intermediate_data_dir)
  agatha_install_path = Path(agatha_install_path)
  if not database_path.is_file():
    if not intermediate_data_dir.exists():
      intermediate_data_dir.mkdir(parents=True, exist_ok=True)
    else:
      # Remove any previously constructed json files in there
      print("\t- Removing existing json files from", intermediate_data_dir)
      for json_file in intermediate_data_dir.glob("*.json"):
        json_file.unlink()

    print("\t- Writing intermediate json files")
    ( # Save all keys and values as kv pair json files
        record_bag
        .map(
          _record_to_kv_json,
          key_field=key_field,
          value_field=value_field
        )
        .to_textfiles(f"{intermediate_data_dir}/*.json")
    )
    print("\t- Writing", database_path)
    _make_sqlite3_database_from_json(
        intermediate_data_dir=intermediate_data_dir,
        database_path=database_path,
        agatha_install_path=agatha_install_path
    )


class Sqlite3LookupTable():
  """
  Gets values from an Sqlite3 Table called "lookup_table" where keys are
  strings and values are json encoded.
  """
  def __init__(self, db_path:Path):
    self.db_path = Path(db_path)
    self.db_conn = None
    self.db_cursor = None
    self.db_cache = {}

  def _cache(self, key:str)->None:
    if key not in self.db_cache:
      assert self.db_cursor is not None, "_cache called outside of with"
      res = self.db_cursor.execute(
          """
          SELECT
            value
          FROM
            lookup_table
          WHERE
            key=?
          ;
          """,
          (key,)
      ).fetchone()
      if res is None:
        self.db_cache[key] = None
      else:
        self.db_cache[key] = json.loads(res[0])

  def __contains__(self, key:str)->bool:
    self._cache(key)
    return self.db_cache[key] is not None

  def __getitem__(self, key:str)->Any:
    self._cache(key)
    return self.db_cache[key]

  def __enter__(self):
    self.db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self.db_conn.execute('PRAGMA journal_mode = OFF')
    self.db_conn.execute('PRAGMA synchronous = OFF')
    self.db_conn.execute('PRAGMA cache_size = 100000')
    self.db_conn.execute('PRAGMA temp_store = MEMORY')
    self.db_cursor = self.db_conn.cursor()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    self.db_conn = None
    self.db_cursor = None
    return False

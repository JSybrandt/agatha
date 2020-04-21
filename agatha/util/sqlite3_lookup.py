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
  print("Sqlite3 Lookup Table:", database_path)
  if database_path.is_file():
    print("\t- Ready")
  else:
    print("\t-Constructing...")
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
    print("\t- Done!")


_DEFAULT_TABLE_NAME="lookup_table"
_DEFAULT_KEY_COLUMN_NAME="key"
_DEFAULT_VALUE_COLUMN_NAME="value"
class Sqlite3LookupTable():
  """
  Gets values from an Sqlite3 Table called where keys are
  strings and values are json encoded.
  """
  def __init__(
      self,
      db_path:Path,
      table_name:str=_DEFAULT_TABLE_NAME,
      key_column_name:str=_DEFAULT_KEY_COLUMN_NAME,
      value_column_name:str=_DEFAULT_VALUE_COLUMN_NAME,
  ):
    self.table_name = table_name
    self.key_column_name = key_column_name
    self.value_column_name = value_column_name
    assert self.key_column_name != self.value_column_name, \
        "Key and Value column names cannot be the same."
    self._connection = None
    self._cursor = None
    db_path = Path(db_path)
    assert db_path.is_file(), f"Failed to find {db_path}"
    self.db_path = Path(db_path)
    self._connect()

  def __del__(self):
    self._disconnect()

  def __getstate__(self):
    self._disconnect()
    state = self.__dict__.copy()
    self._connect()
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._connect()

  def is_preloaded(self)->bool:
    if not self.connected():
      return False
    """
    From: https://www.sqlite.org/pragma.html#pragma_database_list
    This pragma works like a query to return one row for each database
    attached to the current database connection. The second column is "main"
    for the main database file, "temp" for the database file used to store
    TEMP objects, or the name of the ATTACHed database for other database
    files. The third column is the name of the database file itself, or an
    empty string if the database is not associated with a file.
    """
    return self._cursor.execute("PRAGMA database_list").fetchone()[2] == ""

  def preload(self)->None:
    """
    Copies the content of the database to memory
    """
    assert self.connected()
    if not self.is_preloaded():
      memory_db = sqlite3.connect(":memory:")
      self._connection.backup(memory_db)
      self._connection.close()
      self._connection = memory_db
      self._cursor = self._connection.cursor()
      self._set_db_flags()

  def connected(self)->bool:
    return self._connection is not None

  def _assert_schema(self)->None:
    assert self.connected()
    """
    https://www.sqlite.org/pragma.html#table_info
    This pragma returns one row for each column in the named table. Columns in
    the result set include the column name, data type, whether or not the
    column can be NULL, and the default value for the column. The "pk" column
    in the result set is zero for columns that are not part of the primary key,
    and is the index of the column in the primary key for columns that are part
    of the primary key.
    """
    schema_data = self._cursor.execute(
        f"PRAGMA table_info('{self.table_name}')"
    ).fetchall()
    assert len(schema_data) > 0, \
        f"Missing `{self.table_name}` in {self.db_path}"
    assert len(schema_data) >= 2, \
        f"Missing key and/or value column in {self.db_path}"
    schema = {}
    for row in schema_data:
      col_name, dtype = row[1], row[2]
      schema[col_name] = dtype

    assert (
        self.key_column_name in schema
        and schema[self.key_column_name] == "TEXT"
    ), f"Schema missing {self.key_column_name} in {self.db_path}"
    assert (
        self.value_column_name in schema
        and schema[self.value_column_name] == "TEXT"
    ), f"Schema missing {self.value_column_name} in {self.db_path}"

  def _set_db_flags(self)->None:
    assert self.connected()
    assert self._cursor is not None
    self._cursor.execute('PRAGMA journal_mode = OFF')
    self._cursor.execute('PRAGMA synchronous = OFF')
    # Negative cache size indicates kb
    self._cursor.execute('PRAGMA cache_size = -1000000')
    self._cursor.execute('PRAGMA temp_store = MEMORY')
    self._cursor.execute('PRAGMA query_only = TRUE')
    self._cursor.execute('PRAGMA threads = 4')

  def _disconnect(self)->None:
    if self.connected():
      self._connection.close()
      self._cursor = None
      self._connection = None

  def _connect(self)->None:
    """
    Sets self._connection and self._cursor and self.connected = True
    """
    assert self.db_path.is_file(), f"Missing database file: {self.db_path}"
    # close if already open
    self._disconnect()
    # Open read-only connection
    self._connection = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self._cursor = self._connection.cursor()
    try:
      self._assert_schema()
    except AssertionError:
      # Backwards compatibility fallback
      self.table_name=_DEFAULT_TABLE_NAME
      self.key_column_name=_DEFAULT_KEY_COLUMN_NAME
      self.value_column_name=_DEFAULT_VALUE_COLUMN_NAME
      self._assert_schema()
    self._set_db_flags()

  def __getitem__(self, key:str):
    assert self.connected(), "Attempting to get item from closed db."
    res = self._cursor.execute(
        f"""
          SELECT {self.value_column_name}
          FROM {self.table_name}
          WHERE {self.key_column_name}=?
        """,
        (key,)
    ).fetchone()
    if res is None:
      return res
    else:
      return json.loads(res[0])

  def __contains__(self, key:str)->bool:
    assert self.connected(), "Attempting to operate on closed db."
    res = self._cursor.execute(
        f"""
          SELECT EXISTS(
            SELECT 1
            FROM {self.table_name}
            WHERE {self.key_column_name}=?
          )
        """,
        (key,)
    ).fetchone()
    return res[0] == 1

################################################################################
## SPECIAL CASES ###############################################################
################################################################################

# These special cases are added for backwards compatibility. Custom table, key
# and column names are potentially used on old data sources.

class Sqlite3Bow(Sqlite3LookupTable):
  def __init__(
      self,
      db_path:Path,
      table_name:str="sentences",
      key_column_name:str="id",
      value_column_name:str="bow",
  ):
    Sqlite3LookupTable.__init__(
        self,
        db_path=db_path,
        table_name=table_name,
        key_column_name=key_column_name,
        value_column_name=value_column_name,
    )

class Sqlite3Graph(Sqlite3LookupTable):
  def __init__(
      self,
      db_path:Path,
      table_name:str="graph",
      key_column_name:str="node",
      value_column_name:str="neighbors",
  ):
    Sqlite3LookupTable.__init__(
        self,
        db_path=db_path,
        table_name=table_name,
        key_column_name=key_column_name,
        value_column_name=value_column_name,
    )

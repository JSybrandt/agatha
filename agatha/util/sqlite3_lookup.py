import sqlite3
from pathlib import Path
import json
from typing import List, Any, Set, Optional
import dask.bag as dbag
from agatha.util.misc_util import Record
import os

def _record_to_kv_json(
    record:Record,
)->str:
  assert "key" in record
  assert "value" in record
  return json.dumps({
    "key": str(record["key"]),
    "value": record["value"]
  })


def compile_kv_json_dir_to_sqlite3(
    json_data_dir:Path,
    result_database_path:Path,
    agatha_install_path:Path,
    merge_duplicates:bool,
    verbose:bool,
)->None:
  """Merges all key/value json entries into an indexed sqlite3 table

  This function assumes that `json_dir` contains many *.json files. Each file
  should contain one json object per line. Each object should contain a "key"
  and a "value" field. This function will use the c++ `create_lookup_table` by
  executing a subprocess.

  Args:
    json_data_dir: The location containing *.jso. files.
    result_database_path: The location to store the result sqlite3 db.
    agatha_install_path: The location containing the "tools" directory, where
      `create_lookup_table` has been built.
    merge_duplicates: The create_lookup_table utility has two modes. If
      merge_duplicates is False, then we assume there are no key collisions and
      each value is stored as-is. If True, then we combine values associated
      with duplicate keys into arrays of unique elements.
    verbose: If set, print intermediate output of create_lookup_table.

  """
  json_data_dir = Path(json_data_dir)
  result_database_path = Path(result_database_path)
  agatha_install_path = Path(agatha_install_path)

  create_lookup_table_exec = (
      agatha_install_path
      .joinpath("tools")
      .joinpath("create_lookup_table")
      .joinpath("create_lookup_table")
  )
  assert create_lookup_table_exec.is_file(), \
      "Failed to find create_lookup_table tool. Incorrect install path?"
  assert json_data_dir.is_dir()
  assert not result_database_path.exists(), \
      f"Database: {result_database_path} already exists."
  flag = os.system(" ".join([
      str(create_lookup_table_exec),
      "-i", str(json_data_dir),
      "-o", str(result_database_path),
      ("-m" if merge_duplicates else ""),
      ("-v" if verbose else ""),
  ]))
  assert flag == 0, "Something failed during create_lookup_table"
  assert result_database_path.is_file(), "Failed to create database"


def export_key_value_records(
    key_value_records:dbag.Bag,
    export_dir:Path,
)->None:
  """Converts a Dask bag of Dicts into a collection of json files.

  In order to create a lookup table, we must first export all data as json.
  This function maps each element of the input bag to a json encoded string and
  writes one file per partition to the export_dir. WARNING: this function will
  delete any json files already present in export_dir.

  Args:
    key_value_records: A dask bag containing dicts.
    export_dir: The location to write json files. Will erase any if present
      beforehand.

  """
  export_dir = Path(export_dir)
  # Clean up / setup export dir
  export_dir.mkdir(parents=True, exist_ok=True)
  # Remove any previously constructed json files in there
  for json_file in export_dir.glob("*.json"):
    json_file.unlink()
  (
    key_value_records
    .map(_record_to_kv_json)
    .to_textfiles(f"{export_dir}/*.json")
  )


def create_lookup_table(
    key_value_records:dbag.Bag,
    result_database_path:Path,
    intermediate_data_dir:Path,
    agatha_install_path:Path,
    merge_duplicates:bool=False,
    verbose:bool=False
)->None:
  """Creates an Sqlite3 table compatible with Sqlite3LookupTable

  Each element of the key_value_records bag is converted to json and written to
  disk. Then, one machine calls the `create_lookup_table` tool in order to
  index all records into an Sqlite3LookupTable compatible database. Warning, if
  used in a distributed setting, the master node will be the one to call the
  `create_lookup_table` utility.

  key_value_records: A dask bag containing dicts. Each dict should have a "key"
    and a "value" field.
  result_database_path: The location to write the Sqlite3 file.
  intermediate_data_dir: The location to write intermediate json text files.
    Warning, if any json files exist beforehand, they will be erased.
  agatha_install_path: The root of Agatha, wherein the `tools` directory can be
    located.
  merge_duplicates: If set, `create_lookup_table` will perform the more
    expensive operation of combining distinct values associated with the same
    key.
  verbose: If set, the `create_lookup_table` utility will print intermediate
    output.

  """
  result_database_path = Path(result_database_path)
  intermediate_data_dir = Path(intermediate_data_dir)
  agatha_install_path = Path(agatha_install_path)

  print("Sqlite3 Lookup Table:", result_database_path)
  if result_database_path.is_file():
    print("\t- Ready")
  else:
    print("\t- Exporting to json")
    export_key_value_records(key_value_records, intermediate_data_dir)
    print("\t- Indexing database")
    compile_kv_json_dir_to_sqlite3(
        json_data_dir=intermediate_data_dir,
        result_database_path=result_database_path,
        agatha_install_path=agatha_install_path,
        merge_duplicates=merge_duplicates,
        verbose=verbose,
    )
    print("\t- Done!")


################################################################################
# Actual Database Interface ####################################################
################################################################################

_DEFAULT_TABLE_NAME="lookup_table"
_DEFAULT_KEY_COLUMN_NAME="key"
_DEFAULT_VALUE_COLUMN_NAME="value"
class Sqlite3LookupTable():
  """Dict-like interface for Sqlite3 key-value tables

  Assumes that the provided sqlite3 path has a table containing string keys and
  json-encoded string values. By default, the table name is
  `lookup_table`, with columns `key` and `value`.

  This interface is pickle-able, and provides caching and preloading. Note that
  instances of this object that are recovered from pickles will _NOT_ retain the
  preloading or caching information from the original.

  Args:
    db_path: The file-system location of the Sqlite3 file.
    table_name: The sql table name to find within `db_path`.
    key_column_name: The string column of `table_name`. Performance of the
      Sqlite3LookupTable will depend on whether an index has been created on
      `key_column_name`.
    value_column_name: The json-encoded string column of `table_name`
    disable_cache: If set, objects resulted from json parsing will not be cached

  """
  def __init__(
      self,
      db_path:Path,
      table_name:str=_DEFAULT_TABLE_NAME,
      key_column_name:str=_DEFAULT_KEY_COLUMN_NAME,
      value_column_name:str=_DEFAULT_VALUE_COLUMN_NAME,
      disable_cache:bool=False
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
    self.db_path = db_path
    self._connect()
    self._use_cache = not disable_cache
    self._cache = {}
    self._len = None # Saved after first len() call

  def __del__(self):
    self._disconnect()

  def __getstate__(self):
    self._disconnect()
    cached_data = self._cache
    self._cache = {}
    state = self.__dict__.copy()
    self._connect()
    self._cache = cached_data
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self._connect()

  def is_preloaded(self)->bool:
    "True if database has been loaded to memory."
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
    """Copies the database to memory.

    This is done by dumping the contents of disk into ram, and _does not_
    perform any json parsing. This improves performance because now sqlite3
    calls do not have to travel to storage.

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
    "True if the database connection has been made."
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
    self._cursor.arraysize = 64

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

  def clear_cache(self)->None:
    "Removes contents of internal cache"
    self._cache.clear()

  def disable_cache(self)->None:
    "Disables the use of internal cache"
    self.clear_cache()
    self._use_cache = False

  def enable_cache(self)->None:
    "Enables the use of internal cache"
    self._use_cache = True

  def _query(self, key:str)->Optional[Any]:
    assert self.connected(), "Attempting to query item from closed db."
    res = self._cursor.execute(
        f"""
          SELECT {self.value_column_name}
          FROM {self.table_name}
          WHERE {self.key_column_name}=?
        """,
        (key,)
    ).fetchone()
    if res is None:
      return None
    else:
      return json.loads(res[0])

  def _get(self, key:str)->Optional[Any]:
    if self._use_cache and key in self._cache:
      return self._cache[key]
    else:
      value = self._query(key)
      if self._use_cache:
        self._cache[key] = value
      return value

  def __getitem__(self, key:str):
    value_or_none = self._get(key)
    assert value_or_none is not None, \
        f"Failed to find {key} in {self.db_path}"
    return value_or_none

  def __contains__(self, key:str)->bool:
    value_or_none = self._get(key)
    return value_or_none is not None

  def keys(self)->Set[str]:
    """Get all keys from the Sqlite3 Table.

    Recalls _all_ keys from the connected database. This operation may be slow
    or even infeasible for larger tables.

    Returns:
      The set of all keys from the connected database.
    """
    assert self.connected(), "Attempting to operate on closed db."
    query = self._cursor.execute(
        f"""
          SELECT {self.key_column_name}
          FROM {self.table_name}
        """
    )
    return set(r[0] for r in query.fetchall())

  def __len__(self)->int:
    "Returns the number of entries in the connected database."
    if self._len is None:
      assert self.connected(), "Attempting to operate on closed db."
      self._len = self._cursor.execute(
          f"""
            SELECT count(*)
            FROM {self.table_name}
          """
      ).fetchone()[0]
    return self._len


################################################################################
## SPECIAL CASES ###############################################################
################################################################################

class Sqlite3Bow(Sqlite3LookupTable):
  """
  For backwards compatibility, Sqlite3Bow allows for alternate default table,
  key, and value names. However, newer tables following the default
  Sqlite3LookupTable schema will still work.
  """
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
        **kwargs
    )

class Sqlite3Graph(Sqlite3LookupTable):
  """
  For backwards compatibility, Sqlite3Graph allows for alternate default table,
  key, and value names. However, newer tables following the default
  Sqlite3LookupTable schema will still work.
  """
  def __init__(
      self,
      db_path:Path,
      table_name:str="graph",
      key_column_name:str="node",
      value_column_name:str="neighbors",
      **kwargs
  ):
    Sqlite3LookupTable.__init__(
        self,
        db_path=db_path,
        table_name=table_name,
        key_column_name=key_column_name,
        value_column_name=value_column_name,
        **kwargs
    )

import sqlite3
from pathlib import Path
import json
from typing import Set
from math import log2
from pymoliere.util.misc_util import iter_to_batches

class Sqlite3Graph(object):
  def __init__(self, db_path:Path):
    db_path = Path(db_path)
    assert db_path.is_file(), "Sqlite database not found."
    "A sqlite3 graph has a db caled 'graph' and columns: node and neighbors'"
    self.select_neighbors_stmt = """
      SELECT
        neighbors
      FROM
        graph
      WHERE
        node=?
      ;
    """
    self.node_exists_stmt = """
    SELECT
      EXISTS(
        SELECT
          1
        FROM
          graph
        WHERE
          node=?
      )
    ;
    """
    self.db_path = db_path
    self.db_conn = None
    self.db_cursor = None
    self._contains_cache = {}
    self._item_cache = {}

  def __contains__(self, entity:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    cache_res = self._contains_cache.get(entity)
    if cache_res is None:
      cache_res = self._contains_cache[entity] = (
          self.db_cursor.execute(
            self.node_exists_stmt,
            (entity,)
          )
          .fetchone()[0]
          == 1  # EXISTS returns 0 or 1
      )
    return cache_res

  def __getitem__(self, entity:str)->Set[str]:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    cache_res = self._item_cache.get(entity)
    if cache_res is None:
      cache_res = self._item_cache[entity] = set(
          json.loads(
            self.db_cursor.execute(
              self.select_neighbors_stmt,
              (entity,)
            )
            .fetchone()[0]
        )
      )
    return cache_res

  def _config_connection(self)->None:
    self.db_conn.execute('PRAGMA journal_mode = OFF')
    self.db_conn.execute('PRAGMA synchronous = OFF')
    self.db_conn.execute('PRAGMA cache_size = 100000')
    self.db_conn.execute('PRAGMA temp_store = MEMORY')
    self.db_cursor = self.db_conn.cursor()

  def __enter__(self):
    self.db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self._config_connection()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    self.db_conn = None
    self.db_cursor = None
    return False

  def weight(self, source:str, target:str)->float:
    return log2(len(self[source])) * log2(len(self[target]))

class PreloadedSqlite3Graph(Sqlite3Graph):
  def __init__(self, db_path:Path):
    super(PreloadedSqlite3Graph, self).__init__(db_path)

  def __enter__(self):
    file_db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self.db_conn = sqlite3.connect(":memory:")
    file_db_conn.backup(self.db_conn)
    file_db_conn.close()
    self._config_connection()
    return self

import sqlite3
from pathlib import Path
import json
from typing import Set
from math import log2

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
    self.db_path = db_path
    self.db_conn = None
    self.db_cursor = None
    self._cache = {}

  def __contains__(self, entity:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    res = self._cache.get(entity, "Missing")
    if res == "Missing":
      dl = (
          self.db_cursor
          .execute(self.select_neighbors_stmt, (entity,))
          .fetchone()
      )
      if dl is not None:
        dl = set(json.loads(dl[0]))
      res = self._cache[entity] = dl
    return res is not None

  def __getitem__(self, entity:str)->Set[str]:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    assert entity in self, f"Failed to find {entity}"
    return self._cache[entity]

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
    self.graph_data = {}

  def __contains__(self, entity:str)->bool:
    return entity in self.graph_data

  def __getitem__(self, entity:str)->Set[str]:
    return self.graph_data[entity]

  def __enter__(self):
    print("Loading SqliteDB")
    file_db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self.graph_data = {
        node: json.loads(neighbors)
        for node, neighbors in
        file_db_conn.execute("SELECT node, neighbors FROM graph;").fetchall()
    }
    file_db_conn.close()
    return self

  def __exit__(self, *args, **kwargs):
    self.graph_data = {}
    return False


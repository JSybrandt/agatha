import sqlite3
from pathlib import Path
import json
from typing import List


class Sqlite3Graph():
  def __init__(self, db_path:Path):
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

  def __contains__(self, entity:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    return (
        self.db_cursor.execute(
          self.node_exists_stmt,
          (entity,)
        )
        .fetchone()[0]
        == 1  # EXISTS returns 0 or 1
    )

  def __getitem__(self, entity:str)->List[str]:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    return json.loads(
        self.db_cursor.execute(
          self.select_neighbors_stmt,
          (entity,)
        )
        .fetchone()[0]
    )

  def __enter__(self):
    self.db_conn = sqlite3.connect(self.db_path)
    self.db_cursor = self.db_conn.cursor()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    self.db_conn = None
    self.db_cursor = None
    return False


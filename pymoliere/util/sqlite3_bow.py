import sqlite3
from pathlib import Path
import json
from typing import List

class Sqlite3Bow():
  def __init__(self, db_path:Path):
    db_path = Path(db_path)
    assert db_path.is_file(), "Sqlite database not found."
    "A sqlite3 graph has a db caled 'sentences' and columns: id and bow'"
    self.select_bow_stmt = """
      SELECT
        bow
      FROM
        sentences
      WHERE
        id=?
      ;
    """
    self.id_exists_stmt = """
    SELECT
      EXISTS(
        SELECT
          1
        FROM
          sentences
        WHERE
          id=?
      )
    ;
    """
    self.db_path = db_path
    self.db_conn = None
    self.db_cursor = None

  def __contains__(self, id_:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    return (
        self.db_cursor.execute(
          self.id_exists_stmt,
          (id_,)
        )
        .fetchone()[0]
        == 1  # EXISTS returns 0 or 1
    )

  def __getitem__(self, id_:str)->List[str]:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    return json.loads(
        self.db_cursor.execute(
          self.select_bow_stmt,
          (id_,)
        )
        .fetchone()[0]
    )

  def __enter__(self):
    self.db_conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    self.db_cursor = self.db_conn.cursor()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    self.db_conn = None
    self.db_cursor = None
    return False


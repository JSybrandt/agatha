import sqlite3
from pathlib import Path
import json
from typing import List, Optional

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
    self.db_cache = {}

  def _get_result_or_none(self, id_:str)->Optional[List[str]]:
    res = self.db_cursor.execute(self.select_bow_stmt, (id_,)).fetchone()
    if res is None:
      return res
    else:
      return json.loads(res[0])

  def __contains__(self, id_:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    res = self.db_cache.get(id_, "Missing")
    if res == "Missing":
      # cache result
      res = self.db_cache[id_] = self._get_result_or_none(id_)
    return res is not None

  def __getitem__(self, id_:str)->List[str]:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    assert id_ in self # side effect of loading value
    return self.db_cache[id_]

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


from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass
from copy import copy
import json
import numpy as np
import h5py
from tqdm import tqdm
import sqlite3

@dataclass
class EmbeddingLocation:
  "This class stores information needed to load an embedding"
  entity_type: str
  partition_idx: int
  row_idx: Optional[int] = None

class EmbeddingLocationIndex(object):
  """
  This class manages pulling embedding locations from the sqlite3 database
  """
  def __init__(self, db_path:Path, db_name:str="embedding_locations"):
    db_path = Path(db_path)
    assert db_path.is_file(), "Invalid path to database."
    self.db_path = db_path
    self.db_name = db_name
    self.db_conn = None
    self.db_cursor = None
    self.select_fmt_str = """
      SELECT
        entity_type,
        partition_idx,
        row_idx
      FROM
        {db_name}
      WHERE
        entity=?
      ;
    """
    self.exists_fmt_str = """
    SELECT
      EXISTS(
        SELECT
          1
        FROM
          {db_name}
        WHERE
          entity=?
      )
    ;
    """

  def __contains__(self, entity:str)->bool:
    assert self.db_cursor is not None, "__contains__ called outside of with"
    return (
        self.db_cursor.execute(
          self.exists_fmt_str.format(db_name=self.db_name),
          entity
        )
        .fetchone()[0]
        == 1  # EXISTS returns 0 or 1
    )

  def __getitem__(self, entity:str)->EmbeddingLocation:
    assert self.db_cursor is not None, "__getitem__ called outside of with"
    return (
        self.db_cursor.execute(
          self.select_fmt_str.format(db_name=self.db_name),
          (entity,)
        )
        .fetchone()
    )

  def _query_to_emb_loc(self, cursor, row)->EmbeddingLocation:
    if len(row) == 3:
      return EmbeddingLocation(*row)
    else:
      return row

  def __enter__(self):
    self.db_conn = sqlite3.connect(self.db_path)
    self.db_conn.row_factory = self._query_to_emb_loc
    self.db_cursor = self.db_conn.cursor()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.db_conn.close()
    return False

class EmbeddingIndex(object):
  def __init__(
      self,
      embedding_dir:Path,
      emb_loc_db_path:Path,
      emb_ver:str=None
  ):
    embedding_dir = Path(embedding_dir)
    emb_loc_db_path = Path(emb_loc_db_path)
    # Setup entity->location index index
    self.embedding_location_index = EmbeddingLocationIndex(emb_loc_db_path)
    self.inside_context_mngr = False

    # This dir holds embedding files
    self.embedding_dir = Path(embedding_dir)
    assert self.embedding_dir.is_dir(), "Invalid embedding_dir"

    # Setup embedding version
    valid_emb_ver = self.load_embedding_versions(embedding_dir)
    assert len(valid_emb_ver) > 0, "Invalid embedding dir, has no versions."
    if emb_ver is None:
      assert len(valid_emb_ver) == 1, \
        f"Must supply emb_ver if multiple exist: {valid_emb_ver}"
      self.emb_ver = next(iter(valid_emb_ver))
    else:
      assert emb_ver in valid_emb_ver, "Invalid emb_ver"
      self.emb_ver = emb_ver

  @staticmethod
  def load_embedding_versions(embedding_dir:Path)->Set[str]:
    return set(map(
      lambda p: p.name.split(".")[1],
      embedding_dir.glob("embeddings_*.v*.h5")
    ))

  def __enter__(self):
    self.inside_context_mngr = True
    self.embedding_location_index.__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.inside_context_mngr = False
    self.embedding_location_index.__exit__(exc_type, exc_value, traceback)
    return False # Don't want to handle exceptions

  def __contains__(self, name:str)->bool:
    assert self.inside_context_mngr, "Called __contains__ outside of with"
    return name in self.embedding_location_index

  def __getitem__(self, name:str)->np.array:
    assert self.inside_context_mngr, "Called __getitem__ outside of with"
    emb_loc = self.embedding_location_index[name]
    assert emb_loc is not None, f"EmbeddingIndex does not contain {name}"
    return self._load_embedding_from_h5(emb_loc)

  def _load_embedding_from_h5(self, emb_loc:EmbeddingLocation)->np.array:
    h5_path = self._get_embedding_path(emb_loc)
    assert h5_path.is_file(), f"{emb_loc} -> {h5_path} Does not exist."
    with h5py.File(h5_path, "r") as h5_file:
      return h5_file["embeddings"][emb_loc.row_idx]

  def _get_embedding_path(self, el:EmbeddingLocation)->Path:
    return self.embedding_dir.joinpath(
        f"embeddings_{el.entity_type}_{el.partition_idx}.{self.emb_ver}.h5"
    )

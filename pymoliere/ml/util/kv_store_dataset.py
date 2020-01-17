from pathlib import Path
from sqlitedict  import SqliteDict
from torch.utils.data import Dataset
from typing import List
from itertools import chain
from bisect import bisect_right

def get_sqlite_files(base_dir:Path)->List[Path]:
  return sorted(
      list(
        chain(
          base_dir.glob("*.sqlite"),
          base_dir.glob("*.sqlitedict")
        )
      )
  )

class KVStoreDictDataset(Dataset):
  def __init__(self, db_dir:Path):
    super(KVStoreDictDataset, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = get_sqlite_files(db_dir)
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    self.sizes = [0]
    for p in self.db_paths:
      with SqliteDict(p, flag='r') as conn:
        self.sizes.append(self.sizes[-1] + len(conn))
    self.connections = {}

  def __len__(self):
    return self.sizes[-1]

  def __getitem__(self, data_idx):
    assert data_idx >= 0
    assert data_idx < len(self)
    # bisect-right is currently the rightmost value greater than global_idx
    # so that - 1 is the index of the cell we want in
    # because of the above asserts, we're good.
    db_idx = bisect_right(self.sizes, data_idx) - 1
    local_idx = data_idx - self.sizes[db_idx]
    if db_idx not in self.connections:
      self.connections[db_idx] = SqliteDict(self.db_paths[db_idx], flag='r')
    try:
      return self.connections[db_idx][local_idx]
    except KeyError as ke:
      print("FAILED TO GET", ke)
      print(db_idx)
      print(local_idx)
      print(len(self.connections[db_idx]))
      print(self.db_paths[db_idx])
      raise ke

class LoadWholeKVStore(Dataset):
  def __init__(self, db_dir):
    super(LoadWholeKVStore, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = get_sqlite_files(db_dir)
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    with SqliteDict(self.db_paths[0], flag='r') as conn:
      self.data = list(conn.values())

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


from pathlib import Path
from sqlitedict  import SqliteDict
from torch.utils.data import Dataset

class KVStoreDictDataset(Dataset):
  def __init__(self, db_dir:Path):
    super(KVStoreDictDataset, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = sorted(list(db_dir.glob("*.sqlite")))
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    self.sizes = [0]
    for p in self.db_paths:
      with SqliteDict(p, flag='r') as conn:
        self.sizes.append(self.sizes[-1] + len(conn))
    self.connections = {}

  def __len__(self):
    return self.sizes[-1]

  def __getitem__(self, data_idx):
    for db_idx in range(len(self.db_paths)):
      if self.sizes[db_idx] <= data_idx < self.sizes[db_idx+1]:
        local_idx = data_idx - self.sizes[db_idx]
        if db_idx not in self.connections:
          self.connections[db_idx] = SqliteDict(self.db_paths[db_idx], flag='r')
        return self.connections[db_idx][str(local_idx)]

class LoadWholeKVStore(Dataset):
  def __init__(self, db_dir):
    super(LoadWholeKVStore, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = sorted(list(db_dir.glob("*.sqlite")))
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    with SqliteDict(self.db_paths[0], flag='r') as conn:
      self.data = list(conn.values())

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


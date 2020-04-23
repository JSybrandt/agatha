from pathlib import Path
from torch.utils.data import Dataset
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
from typing import Tuple, Any

class Sqlite3Dataset(Dataset):
  def __init__(self, table:Sqlite3LookupTable):
    self.table=table
    self._keys = None
    self._len = None

  def __getstate__(self):
    "Don't serialize the keys."
    keys = self._keys
    self._keys = None
    state = self.__dict__.copy()
    self._keys = keys
    return state

  def __len__(self):
    if self._len is None:
      self._len = len(self.table)
    return self._len

  def __getitem__(self, idx:int)->Tuple[str, Any]:
    assert 0 <= idx < len(self), "Invalid index."
    if self._keys is None:
      self._keys = list(self.table.keys())
    key = self._keys[idx]
    value = self.table[key]
    return key, value

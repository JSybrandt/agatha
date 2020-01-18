import json
from typing import List
from pathlib import Path
from bisect import bisect_right

class EntityIndex(object):
  def __init__(self, entity_dir:Path, entity_type:str):
    entity_dir = Path(entity_dir)
    assert entity_dir.is_dir()

    self.count_paths = sorted(list(
      entity_dir.glob(f"entity_count_{entity_type}_*.txt")
    ))
    self.name_paths = sorted(list(
      entity_dir.glob(f"entity_names_{entity_type}_*.json")
    ))
    assert len(self.count_paths) > 0, "Missing count files"
    assert len(self.count_paths) == len(self.name_paths), \
        "Unequal # of name/count files"
    self.prefix_count = self._get_prefix_count(self.count_paths)
    self.path_idx2names = {}

  @staticmethod
  def _get_prefix_count(count_paths:List[Path])->List[int]:
    counts = [0]
    for p in count_paths:
      with open(p) as f:
        this_count = int(f.read())
      counts.append(this_count + counts[-1])
    return counts

  def get_path_idx(self, global_idx:int)->int:
    assert global_idx < len(self)
    assert global_idx >= 0
    # bisect-right is currently the rightmost value greater than global_idx
    # so that - 1 is the index of the cell we want in
    # because of the above asserts, we're good.
    return bisect_right(self.prefix_count, global_idx) - 1

  def __len__(self):
    return self.prefix_count[-1]

  def __getitem__(self, global_idx:int)->str:
    path_idx = self.get_path_idx(global_idx)
    assert path_idx is not None
    if path_idx not in self.path_idx2names:
      with open(self.name_paths[path_idx]) as f:
        self.path_idx2names[path_idx] = json.load(f)
    return self.path_idx2names[path_idx][global_idx-self.prefix_count[path_idx]]



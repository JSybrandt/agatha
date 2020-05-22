#!/usr/bin/env python3
"""
Converts SemMedDB CSV for input into Edge2Vec

Edge2Vec input format is <source id> <target id> <relation id> <edge id>

For example
```
1 4 1 1
1 5 1 2
1 6 1 3
1 7 1 4
2 7 1 5
2 8 1 6
2 9 1 7
2 10 1 8
```

"""

from fire import Fire
from pathlib import Path
from agatha.util import semmeddb_util as sm
from agatha.util import entity_types as typs
import pickle
from itertools import product
from typing import Dict

def get_or_add_idx(name:str, name2idx:Dict[str, int])->int:
  # Either int or None
  idx = name2idx.get(name)
  if idx is None:
    # ids should start at 1
    idx = name2idx[name] = len(name2idx) + 1
  return idx

def main(
  semmeddb_csv_path:Path,
  output_edge_list:Path,
  output_index:Path,
  cut_date:sm.Datestr=None,
):
  semmeddb_csv_path = Path(semmeddb_csv_path)
  output_edge_list = Path(output_edge_list)
  output_index = Path(output_index)
  assert semmeddb_csv_path.is_file()
  assert not output_edge_list.exists()
  assert not output_index.exists()

  index = dict(
      node2idx={},
      relation2idx={},
  )
  get_or_add_node     = lambda n: get_or_add_idx(n, index["node2idx"])
  get_or_add_relation = lambda n: get_or_add_idx(n, index["relation2idx"])

  predicates = sm.parse(semmeddb_csv_path)
  if cut_date is not None:
    predicates = sm.filter_by_date(predicates, cut_date)

  edge_idx = 1
  with open(output_edge_list, 'w') as out_edge_file:
    for predicate in predicates:
      # Its actually only one ID
      sub = get_or_add_node(predicate["subj_ids"])
      obj = get_or_add_node(predicate["obj_ids"])
      vrb = get_or_add_relation(predicate["pred_type"])
      out_edge_file.write(f"{sub} {obj} {vrb} {edge_idx}\n")
      edge_idx += 1
  with open(output_index, 'wb') as out_index_file:
    pickle.dump(index, out_index_file)


if __name__=="__main__":
  Fire(main)

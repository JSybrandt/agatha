#!/usr/bin/env python3
from fire import Fire
from scipy import sparse
from pathlib import Path
from tqdm import tqdm
from typing import List
import re
import pickle

def to_coo(
  edge_list_tsv:Path,
  out_index:Path,
  out_coo:Path,
):
  edge_list_tsv = Path(edge_list_tsv)
  out_coo = Path(out_coo)
  out_index = Path(out_index)
  assert edge_list_tsv.is_file()
  assert not out_coo.exists() and out_coo.parent.is_dir()
  assert not out_index.exists() and out_index.parent.is_dir()

  name_to_idx = {}

  weights:List[float] = []
  row_indices:List[int] = []
  col_indices:List[int] = []

  # My regex is more resilient (and probably faster) than the csv reader
  line_re = r"(.+)\t(.+)\t([0-9.]+)"
  with open(edge_list_tsv) as csv_file:
    for line in tqdm(csv_file):
      match = re.search(line_re, line)
      if match is not None:
        try:
          s = match.group(1)
          t = match.group(2)
          w = float(match.group(3))
        except:
          print(f"Failed to group '{line}'")
          continue
        row = name_to_idx.setdefault(s, len(name_to_idx))
        col = name_to_idx.setdefault(t, len(name_to_idx))
        row_indices.append(row)
        col_indices.append(col)
        weights.append(w)
      else:
        print(f"Failed to parse '{line}'")
        continue
  with open(out_index, 'wb') as idx_file:
    pickle.dump(name_to_idx, idx_file)

  csr = sparse.coo_matrix(
      (weights, (row_indices, col_indices)),
      shape=(len(name_to_idx), len(name_to_idx))
  )
  sparse.save_npz(str(out_coo), csr)

if __name__ == "__main__":
  Fire(to_coo)


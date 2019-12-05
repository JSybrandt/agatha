#!/usr/bin/env python3
from fire import Fire
from pathlib import Path
import pickle
import dask.bag as dbag
import os
from sqlitedict import SqliteDict

def get_part_size(path:Path)->int:
  with open(path, 'rb') as f:
    return len(pickle.load(f))

def write_part(part_start):
  print("Writing", part_start)
  part, idx = part_start
  with SqliteDict(str(part)+".sqlite") as db, open(part, 'rb') as f:
    for val in pickle.load(f):
      db[str(idx)] = val
      idx += 1
    db.commit()

def write_training_data_to_db(
    part_dir:Path,
):
  part_dir = Path(part_dir)
  assert part_dir.is_dir()

  print("Getting counts")
  part_paths = dbag.from_sequence(
    list(part_dir.glob("*.pkl")),
    npartitions=os.cpu_count()
  )
  part_paths_with_counts = part_paths.map(lambda x: (x, get_part_size(x))).compute()

  print("Getting prefix sum")
  part_paths_with_prefix_sum = []
  idx = 0
  for path, count in part_paths_with_counts:
    part_paths_with_prefix_sum.append((path, idx))
    idx += count

  print("Found", idx, "values")

  print("Writing")
  (
      dbag.from_sequence(part_paths_with_prefix_sum, npartitions=os.cpu_count())
      .map(write_part)
      .compute()
  )

if __name__ == "__main__":
  Fire(write_training_data_to_db)


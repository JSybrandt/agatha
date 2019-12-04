#!/usr/bin/env python3
from fire import Fire
from pathlib import Path
import pickle
import dask.bag as dbag
import os

def count_part(path:Path)->int:
  with open(path, 'rb') as f:
    return len(pickle.load(f))

def count_parts(
    part_dir:Path,
):
  part_dir = Path(part_dir)
  assert part_dir.is_dir()

  lines = (
      dbag.from_sequence(
        list(part_dir.glob("*.pkl")),
        npartitions=os.cpu_count()
      )
      .map(lambda x: f"{x.absolute()} {count_part(x)}")
      .compute()
  )
  for line in lines:
    print(line)


if __name__ == "__main__":
  Fire(count_parts)


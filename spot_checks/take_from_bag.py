#!/usr/bin/env python3
import dask
from pathlib import Path
from pymoliere.util.file_util import load
from argparse import ArgumentParser
from pprint import pprint

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("bag_dir", type=Path)
  parser.add_argument("take", nargs="?", type=int, default=5)
  args = parser.parse_args()
  assert args.bag_dir.is_dir()
  assert args.take > 0
  print(f"Taking {args.take} from {args.bag_dir}")
  pprint(load(args.bag_dir).take(args.take))

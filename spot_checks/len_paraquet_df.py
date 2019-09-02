#!/usr/bin/env python3
from dask.distributed import Client
from argparse import ArgumentParser
from pathlib import Path
from pymoliere.util.file_util import paraquet_exists
import dask.dataframe as ddf

if __name__=="__main__":
  parser = ArgumentParser()
  parser.add_argument("cluster_address", type=str)
  parser.add_argument("path", type=Path)
  parser.add_argument("--sample_rate", type=float, default=0.0001)
  args = parser.parse_args()
  assert paraquet_exists(args.path)
  client = Client(address = args.cluster_address)
  print(args.path.name)
  if args.path.name in client.list_datasets():
    print("\t- Cached")
    recs = client.get_dataset(args.path.name)
  else:
    print("\t- Loading")
    recs = ddf.read_parquet(args.path, engine="pyarrow")
  print(len(recs))

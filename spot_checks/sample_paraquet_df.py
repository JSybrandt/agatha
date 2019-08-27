#!/usr/bin/env python3
from dask.distributed import Client
from argparse import ArgumentParser
from pathlib import Path
from pymoliere.construct.file_util import paraquet_exists
import dask.dataframe as ddf

if __name__=="__main__":
  parser = ArgumentParser()
  parser.add_argument("cluster_address", type=str)
  parser.add_argument("path", type=Path)
  parser.add_argument("--sample_rate", type=float, default=0.00001)
  args = parser.parse_args()
  assert paraquet_exists(args.path)
  client = Client(address = args.cluster_address)
  recs = ddf.read_parquet(args.path, engine="pyarrow")
  vals = recs.sample(frac=args.sample_rate).compute()
  for row in vals.iterrows():
    print(row)

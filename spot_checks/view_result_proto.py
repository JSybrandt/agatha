#!/usr/bin/env python3
from pprint import pprint
from pymoliere.query import query_pb2 as qpb
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("proto", type=Path)
  args = parser.parse_args()
  res = qpb.MoliereResult()
  with open(args.proto, 'rb') as file:
    res.ParseFromString(file.read())
  pprint(res)


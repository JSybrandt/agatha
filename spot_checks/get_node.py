#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
from redis import Redis
from pymoliere.util.db_key_util import GRAPH_TYPE, key_is_type, to_graph_key

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("node_name")
  args = parser.parse_args()
  r = Redis()
  if not key_is_type(args.node_name, GRAPH_TYPE):
    args.node_name = to_graph_key(args.node_name)
  pprint(r.zscan(args.node_name))

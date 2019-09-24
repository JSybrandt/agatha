#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
from redis import Redis
from pymoliere.util.db_key_util import GRAPH_TYPE, key_is_type, to_graph_key

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("node_a")
  parser.add_argument("node_b")
  args = parser.parse_args()
  r = Redis()
  if not key_is_type(args.node_a, GRAPH_TYPE):
    args.node_name = to_graph_key(args.node_a)
  if not key_is_type(args.node_b, GRAPH_TYPE):
    args.node_name = to_graph_key(args.node_b)
  a = r.zscan(args.node_a)[1]
  b = r.zscan(args.node_b)[1]
  a_names = {n for n, _ in a}
  b_names = {n for n, _ in b}
  print(args.node_a)
  pprint(a_names)
  print()
  print(args.node_b)
  pprint(b_names)
  print()
  print("Intersection")
  pprint(a_names.intersection(b_names))

#!/usr/bin/env python3
from argparse import ArgumentParser
from pprint import pprint
from redis import Redis
from pymoliere.util.db_key_util import GRAPH_TYPE, key_is_type, to_graph_key

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("record_name")
  args = parser.parse_args()
  r = Redis()
  assert not key_is_type(args.record_name, GRAPH_TYPE)
  pprint(r.hgetall(args.record_name))

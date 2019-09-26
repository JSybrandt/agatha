#!/usr/bin/env python3
import dask
from argparse import ArgumentParser
from dask.distributed import Client
# import sys
# import importlib

# def reload_modules():
  # module_names = set(sys.modules) & set(globals())
  # all_mods = [sys.modules[name] for name in module_names]
  # for mod in all_mods:
    # importlib.reload(mod)
  # return module_names


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("node")
  args = parser.parse_args()
  client = Client(f"{args.node}:8786")
  client.restart()


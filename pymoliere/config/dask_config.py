from dask.distributed import config
from pathlib import Path

def set_local_tmp(local_dir:Path)->None:
  assert local_dir.is_dir()
  config["temporary-directory"] = str(local_dir)

def print_dask_config()->None:
  print(config)



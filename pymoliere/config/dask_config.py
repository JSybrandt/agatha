from dask.distributed import config
from pathlib import Path

def set_local_tmp(local_dir:Path)->None:
  assert local_dir.is_dir()
  config["temporary-directory"] = str(local_dir)

def set_verbose_worker()->None:
  config["logging"]["worker"] = "info"
  config["logging"]["client"] = "info"

def print_dask_config()->None:
  print(config)



from pathlib import Path
from typing import Callable, Any
import pyarrow as pa

def copy_to_local_scratch(src:Path, local_scratch_dir:Path)->Path:
  local_scratch_dir.mkdir(parents=True, exist_ok=True)
  assert src.is_file()
  assert local_scratch_dir.is_dir()
  dest  = local_scratch_dir.joinpath(src.name)
  dest.write_bytes(src.read_bytes())
  return dest


def prep_scratch_subdir(scratch_root:Path, dir_name:str)->Path:
  assert scratch_root.is_dir()
  res = scratch_root.joinpath(dir_name)
  if not res.exists():
    res.mkdir()
  assert res.is_dir()
  return res


def paraquet_exists(paraquet_dir:Path)->bool:
  "Very simple check"
  if not paraquet_dir.is_dir():
    return False
  return paraquet_dir.joinpath("_metadata").is_file()

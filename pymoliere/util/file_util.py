from pathlib import Path
from typing import Callable, Any, Tuple, Optional
import dask.bag as dbag
import json
import msgpack
from dask.distributed import Lock
import string
import random

EXT = ".json.gz"

def get_random_ascii_str(str_len:int)->str:
  return "".join([random.choice(string.ascii_letters) for _ in range(str_len)])

def copy_to_local_scratch(src:Path, local_scratch_dir:Path)->Path:
  local_scratch_dir.mkdir(parents=True, exist_ok=True)
  assert src.is_file()
  assert local_scratch_dir.is_dir()
  dest  = local_scratch_dir.joinpath(src.name)
  dest.write_bytes(src.read_bytes())
  return dest

def prep_scratches(
    local_scratch_root:Path,
    shared_scratch_root:Path,
    task_name:str,
)->Tuple[Path, Path]:
  local = local_scratch_root.joinpath(task_name)
  shared = shared_scratch_root.joinpath(task_name)
  local.mkdir(parents=True, exist_ok=True)
  shared.mkdir(parents=True, exist_ok=True)
  return local, shared

def load(path:Path)->dbag.Bag:
  assert is_result_saved(path)
  return dbag.read_text(
      str(path.joinpath(f"*{EXT}")),
  ).map(json.loads)

def save(bag:dbag.Bag, path:Path, **kwargs)->None:
  assert path.is_dir()
  return bag.map(json.dumps).to_textfiles(
    path=str(path.joinpath(f"*{EXT}")),
    **kwargs
  )

def is_result_saved(path:Path)->bool:
  files = [
      int(f.name.split(".")[0])
      for f in path.iterdir()
      if f.name.endswith(EXT)
  ]
  files.sort()
  return len(files) > 0 and (files[-1]+1) == len(files)

def touch_random_unused_file(base_dir:Path, ext:Optional[str]=None)->Path:
  assert base_dir.is_dir()
  if ext is None:
    ext = ""
  elif ext[0] != ".":
    ext = f".{ext}"
  lock = Lock(f"dir_lock:{base_dir.name}")
  lock.aquire(timeout=5)
  # THREADSAFE
  name = f"{get_random_ascii_str(10)}{ext}"
  path = base_dir.joinpath(name)
  while path.is_file():
    name = f"{get_random_ascii_str(10)}{ext}"
    path = base_dir.joinpath(name)
  path.touch()
  lock.release()
  return path


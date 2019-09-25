import dask
import dask.bag as dbag
from dask.delayed import Delayed
import pickle
from typing import List
from pymoliere.util.misc_util import Record
from pathlib import Path

_CHECKPOINT_TASKS = []

_CHECKPOINT_NAMES = set()


def get_checkpoint_tasks():
  return _CHECKPOINT_TASKS


def checkpoint(
    data:dbag.Bag,
    name:str,
    checkpoint_dir:Path,
)-> dbag.Bag:
  """
  This function checkpoints a dask bag. The bag is broken into partitions, each
  partition is given a checkpoint task, and then the bag is recombined. Any
  partition that has been previously checkpointed will be restored.
  """
  # Assure ourselves that we have a unique name
  assert name not in _CHECKPOINT_NAMES
  _CHECKPOINT_NAMES.add(name)

  # Setup directory
  assert checkpoint_dir.is_dir()
  part_dir = checkpoint_dir.joinpath(name)
  part_dir.mkdir(parents=True, exist_ok=True)
  assert part_dir.is_dir()

  data_parts = []
  for part_idx, part_data in enumerate(data.to_delayed()):
    part_name = f"part-{part_idx}.pkl"
    part_path = part_dir.joinpath(part_name)
    if part_path.is_file():
      data_parts.append(dask.delayed(read_checkpoint)(part_path))
    else:
      data_parts.append(part_data)
      _CHECKPOINT_TASKS.append(dask.delayed(write_checkpoint)(
        part=part_data,
        part_path=part_path
      ))
  return dbag.from_delayed(data_parts)


def write_checkpoint(part:List[Record], part_path:Path)->None:
  with open(part_path, 'wb') as f:
    pickle.dump(part, f)


def read_checkpoint(part_path:Path)->List[Record]:
  with open(part_path, 'rb') as f:
    return pickle.load(f)

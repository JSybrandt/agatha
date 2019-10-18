import dask.bag as dbag
from pathlib import Path
from pymoliere.construct import file_util

_CHECKPOINT_NAMES = set()


def checkpoint(
    data:dbag.Bag,
    name:str,
    checkpoint_dir:Path,
    respect_partial_checkpoints:bool=True,
    **compute_kwargs,
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

  if file_util.is_result_saved(part_dir):
    print("\t- Cached")
  else:
    file_util.save(
        data,
        part_dir,
        keep_partial_result=respect_partial_checkpoints
    ).compute(**compute_kwargs)
  return file_util.load(part_dir)

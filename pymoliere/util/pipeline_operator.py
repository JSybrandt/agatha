import dask
from dask.delayed import Delayed
import dask.dataframe as ddf
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import shutil
from pymoliere.util import file_util

# Used to remember all existing operator names in a project.
# Produces an error if names are reused.
_OPERATOR_NAMES = set()

DEFAULTS = {
  "local_scratch_root": None,
  "shared_scratch_root": None,
  "clear_scratch": False,
}

class PipelineOperator(ABC):
  def __init__(
      self,
      name:str,
      #-- Replaced with DEFAULTS
      local_scratch_root:Path=None,
      shared_scratch_root:Path=None,
      clear_scratch:bool=None,
  ):
    """
    A pipeline operator maps some input to a distributed dataframe.
    The goal is that any operation can be abstracted to a pipeline operator.
    Arguments:
    - shared_scratch: Location where all operators can access.
    - name: unique name for this particular operator. The results of this
      operator will be recorded in shared_scratch/name
    - clear_scratch: If set, delete the contents of shared_scratch/name
      before computing.
    """
    if local_scratch_root is None:
      local_scratch_root = DEFAULTS.get("local_scratch_root")
    if shared_scratch_root is None:
      shared_scratch_root = DEFAULTS.get("shared_scratch_root")
    if clear_scratch is None:
      clear_scratch = DEFAULTS.get("clear_scratch")

    if name in _OPERATOR_NAMES:
      raise ValueError(f"Duplicate PipelineOperator name detected: {name}.")
    _OPERATOR_NAMES.add(name)
    self.shared_scratch = file_util.prep_scratch_subdir(
        scratch_root=shared_scratch_root,
        dir_name=name,
    )
    # Warning! You need to make local_scratch within the job that uses it.
    self.local_scratch = file_util.prep_scratch_subdir(
        scratch_root=local_scratch_root,
        dir_name=name,
    )
    # Indicates whether the result is written to disk
    self.name = name
    self.result = None
    self.load = None
    self.clear_scratch=clear_scratch

  def is_result_saved(self)->bool:
    return file_util.paraquet_exists(self.shared_scratch)

  def eval(self)->List[Delayed]:
    "Computes and saves, or loads. Either way, persists result and returns it."
    print("Operator:", self.name)
    if self.clear_scratch:
      print("\t- Clearing Scratch")
      self.remove_scratch()
    if self.is_result_saved():
      print("\t- Loading")
      res = self.load()
      tasks = [res]
    else:
      print("\t- Computing")
      res = self.get_dataframe()
      print("\t- Saving")
      tasks = [res, self.save(res)]
    res.persist()
    dask.compute(*tasks)
    return res

  def load(self)->ddf:
    return ddf.read_parquet(self.shared_scratch, engine='pyarrow')

  def save(self, result)->Delayed:
    return result.to_parquet(
      path=self.shared_scratch,
      engine='pyarrow',
      compute=False,
    )

  def remove_scratch(self)->None:
    shutil.rmtree(self.shared_scratch)
    self.shared_scratch.mkdir()
    shutil.rmtree(self.local_scratch)
    self.local_scratch.mkdir()

  def touch_local_scratch(self)->None:
    self.local_scratch.mkdir(parents=True, exist_ok=True)

  @abstractmethod
  def get_dataframe(self)->dask.dataframe.DataFrame:
    "Overwrite this to perform calculations"
    pass

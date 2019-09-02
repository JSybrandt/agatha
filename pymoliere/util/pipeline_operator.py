from __future__ import annotations

import dask
from dask.distributed import Client
from dask.delayed import Delayed
import dask.dataframe as ddf
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import shutil
from pymoliere.util import file_util

# Used to remember all existing operator names in a project.
# Produces an error if names are reused.
_OPERATOR_NAMES = set()

DEFAULTS = {
  "local_scratch_root": None,
  "shared_scratch_root": None,
  "dask_client": None,
}

_PAR_ENGINE="fastparquet" # or pyarrow

class PipelineOperator(ABC):
  def __init__(
      self,
      name:str,
      #-- Truly Optional
      repartition_kwargs:Optional[Dict[str, Any]]=None,
      #-- Replaced with DEFAULTS
      local_scratch_root:Path=None,
      shared_scratch_root:Path=None,
  ):
    """
    A pipeline operator maps some input to a distributed dataframe.
    The goal is that any operation can be abstracted to a pipeline operator.
    Arguments:
    - shared_scratch: Location where all operators can access.
    - name: unique name for this particular operator. The results of this
      operator will be recorded in shared_scratch/name
      before computing.
    - repartition_kwargs: If set, repartition after compute using kwargs
    """
    if local_scratch_root is None:
      local_scratch_root = DEFAULTS.get("local_scratch_root")
    if shared_scratch_root is None:
      shared_scratch_root = DEFAULTS.get("shared_scratch_root")

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
    self.repartition_kwargs=repartition_kwargs
    self.val = None

  def is_result_saved(self)->bool:
    return file_util.paraquet_exists(self.shared_scratch)

  def eval(
      self,
      dask_client:Client=None,
      save=True,
      compute=True
  )->PipelineOperator:
    """
    Either computes, uses cached, or loads. At the end, publishes a dataset
    under 'name'
    Client used to publish dataset. Reads from default if none.
    """
    if dask_client is None:
      dask_client = DEFAULTS.get("dask_client")

    print("Operator:", self.name)

    if self.val is not None:
      print("\t- Already created.")
    elif self.name in dask_client.list_datasets():
      print("\t- Retrieving from published datasets")
      self.val = dask_client.get_dataset(self.name)
    elif self.is_result_saved():
      print("\t- Loading from storage")
      self.val = self.load()
    else:
      print("\t- Computing")
      self.val = self.get_dataframe(dask_client)
      if self.repartition_kwargs is not None:
        print("\t- Repartitioning")
        self.val = self.val.repartition(**self.repartition_kwargs)
    #self.val.persist()
    if self.name not in dask_client.list_datasets():
      print(f"\t- Publishing: '{self.name}'")
      dask_client.publish_dataset(**{self.name: self.val})
    tasks = [self.val]
    if save:
      tasks.append(self.save(compute=False))
    if compute:
      dask_client.compute(*tasks)
    return self

  def load(self)->ddf:
    return ddf.read_parquet(
        self.shared_scratch,
        gather_statistics=False,
        engine=_PAR_ENGINE
    )

  def save(self, compute:bool=True, force=False)->Optional[Delayed]:
    assert self.val is not None
    if not self.is_result_saved() or force:
      print(f"\t- Saving: '{self.shared_scratch}'")
      return self.val.to_parquet(
        path=self.shared_scratch,
        engine=_PAR_ENGINE,
        compute=compute,
      )
    return None

  def persist(self)->PipelineOperator:
    self.val.persist()
    return self

  def unpersist(self)->PipelineOperator:
    del self.val
    self.val = None
    return self

  def remove_scratch(self)->None:
    shutil.rmtree(self.shared_scratch)
    self.shared_scratch.mkdir()
    shutil.rmtree(self.local_scratch)
    self.local_scratch.mkdir()

  def touch_local_scratch(self)->None:
    self.local_scratch.mkdir(parents=True, exist_ok=True)

  @abstractmethod
  def get_dataframe(self, dask_client:Client)->dask.dataframe.DataFrame:
    "Overwrite this to perform calculations"
    pass


class FilterOperator(PipelineOperator):
  def __init__(self, input_dataset:str, expression:str, **kwargs):
    PipelineOperator.__init__(self, **kwargs)
    self.input_dataset=input_dataset
    self.expression = expression

  def get_dataframe(self, dask_client:Client):
    input_data = dask_client.get_dataset(self.input_dataset)
    return input_data.query(self.expression)

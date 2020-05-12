"""A singleton responsible for saving and loading dask bags.
"""

from typing import Dict, Any, Optional, Set
import dask.bag as dbag
from agatha.construct import file_util
from pathlib import Path
from copy import deepcopy
import shutil
import inspect
from datetime import datetime


# GLOBAL PARAMETERS
_PARAM:Dict[str, Any] = dict(
  ckpt_root=None,
  verbose=True,
  allow_partial=True,
  enabled=True,
  halt_after_ckpt=None,
)

# DEFAULT USED TO RESET ACTIVE PARAM
__DEFAULT_PARAM = deepcopy(_PARAM)


def _reset_param()->None:
  for name in _PARAM:
    _PARAM[name] = __DEFAULT_PARAM[name]

def set_halt_point(name:str)->None:
  _PARAM["halt_after_ckpt"] = name

def clear_halt_point()->None:
  _PARAM["halt_after_ckpt"] = None

def disable()->None:
  _PARAM["enabled"] = False

def enable()->None:
  _PARAM["enabled"] = True

def set_allow_partial(allow:bool)->None:
  _PARAM["allow_partial"] = allow

def get_allow_partial()->bool:
  return _PARAM["allow_partial"]

def set_verbose(is_verbose:bool)->None:
  _PARAM["verbose"] = is_verbose

def get_verbose()->bool:
  return _PARAM["verbose"]

def set_root(ckpt_root:Path)->None:
  ckpt_root = Path(ckpt_root)
  ckpt_root.mkdir(parents=True, exist_ok=True)
  assert ckpt_root.is_dir()
  _PARAM["ckpt_root"] = ckpt_root

def get_root()->Path:
  assert "ckpt_root" in _PARAM, "Invalid _PARAM"
  assert _PARAM["ckpt_root"] is not None, "set_root not called!"
  return _PARAM["ckpt_root"]

def get_or_make_ckpt_dir(name:str)->Path:
  ckpt_dir = get_root().joinpath(name)
  ckpt_dir.mkdir(parents=True, exist_ok=True)
  return ckpt_dir

def clear_all_ckpt()->None:
  shutil.rmtree(get_root())
  get_root().mkdir(parents=True, exist_ok=True)

def clear_ckpt(name:str)->None:
  ckpt_dir = get_or_make_ckpt_dir(name)
  shutil.rmtree(ckpt_dir)

def is_ckpt_done(name:str)->bool:
  ckpt_dir = get_or_make_ckpt_dir(name)
  return file_util.is_result_saved(ckpt_dir)

def get_done_file_path(name:str)->Path:
  return get_or_make_ckpt_dir(name).joinpath(file_util.DONE_FILE)

def get_checkpoints_like(glob_pattern:str)->Set[Path]:
  ckpt_names = {
      p.name for p in get_root().glob(glob_pattern) if p.is_dir()
  }
  return {n for n in ckpt_names if is_ckpt_done(n)}

# CHECKPOINT INTERFACE

def checkpoint(
    name:str,
    bag:Optional[dbag.Bag]=None,
    verbose:Optional[bool]=None,
    allow_partial:Optional[bool]=None,
    halt_after:Optional[str]=None,
    textfile:bool=False,
    **compute_kw
)->Optional[dbag.Bag]:
  """Stores the contents of the bag as a series of files.

  This function takes each partition of the input bag and writes them to files
  within a directory associated with the input name. The location of each
  checkpoint directory is dependent on the `ckpt_root` option.

  For each optional argument, (other than `bag`) of this function, there is an
  associated module-level parameter that can be set globally.

  The module-level parameter checkpoint_root, set with `set_root` must be set
  before calling checkpoint.

  Usage:
    checkpoint(name) - returns load opt for checkpoint "name"
    checkpoint(name, bag) - if ckpt
    writes bag to ckpt "name" and returns load op
    if disable() was called, returns the input bag

  Args:
    name: The name of the checkpoint directory to lookup or save to
    bag: If set, save this bag. Otherwise, we will require that this checkpoint
      has already been saved.
    verbose: Print helper info. If unspecified, defaults to module-level parameter.
    allow_partial: If true, partial files present in an unfinished checkpoint
      directory will not be overwritten. If false, unfinished checkpoints will
      be recomputed in full. Defaults to module-level parameter if unset.
    halt_after: If set to the name of the current checkpoint, the agatha process
      will stop after computing its contents. This is important for partial
      pipeline runs, for instance, for computing training data for an ml model.
    textfile: If set, checkpoint will be stored in plaintext format, used to
      save strings. This results in this function returning `None`.

  Returns:
    A dask bag that, if computed, _LOADS_ the specified checkpoint. This means
    that future operations can depend on the loading of intermediate data,
    rather than the intermediate computations themselves.
  """
  if verbose is None:
    verbose = get_verbose()
  if allow_partial is None:
    allow_partial = get_allow_partial()
  if halt_after is None:
    halt_after = (
        _PARAM["halt_after_ckpt"] is not None
        and _PARAM["halt_after_ckpt"] == name
    )

  def vprint(*args, **kwargs):
    if verbose:
      print(*args, **kwargs)

  def check_halt():
    if halt_after:
      vprint("\t- Halting")
      exit(0)

  # If checkpoint is done, load no matter what
  vprint("Checkpoint:", name, "\t", datetime.now())
  if is_ckpt_done(name):
    vprint("\t- Ready")
    check_halt()
    if textfile:
      return None
    else:
      return file_util.load(get_or_make_ckpt_dir(name))

  # If check pointing is enabled, we need to save the bag and return the load fn
  if _PARAM["enabled"]:
    assert bag is not None, f"Checkpoint needs bag argument to load {name}"
    vprint("\t- Saving")
    file_util.save(
        bag=bag,
        path=get_or_make_ckpt_dir(name),
        keep_partial_result=allow_partial,
        textfile=textfile,
    ).compute(**compute_kw)
    vprint("\t- Done!")
    check_halt()
    if textfile:
      return None
    else:
      return file_util.load(get_or_make_ckpt_dir(name))

  # If check pointing is disabled, we just return the in-progress bag.
  else:  #disabled
    assert bag is not None, \
        f"Checkpointing is disabled, and no bag specified for {name}"
    vprint("\t- Checkpoint Disabled")
    check_halt()
    return bag

def ckpt(bag_name:str, ckpt_prefix:Optional[str]=None, **kwargs)->None:
  """Simple checkpoint interface

  This is syntactic sugar for the most common use case.
  You can replace
  ```
  my_dask_bag = checkpoint("my_dask_bag", my_dask_bag)
  ```

  ```
  ckpt("my_dask_bag")
  ```

  Calling this function will replace the variable associated with `bag_name`
  after computing its checkpoint.  This means that calling compute on later
  calls of `bag_name` will load that bag from storage, rather than perform all
  intermediate computations again.

  Args:
    bag_name: The name of a local variable corresponding to a dask bag. This bag
      will be computed and stored to a checkpoint of the same name. The bag
      variable will be replaced with a new bag that can be loaded from this
      checkpoint.
    ckpt_prefix: If set, the provided string will be prefixed to the bag_name
      checkpoint. This allows the same variable names to be associated with
      different checkpoints. For instance, the `document_pipeline` functions
      create a bag named "sentences" regardless of the set of documents used
      to create those sentences. By specifying a prefix, different calls to
      `document_pipeline` can create different checkpoints.

  """

  caller_locals = inspect.currentframe().f_back.f_locals
  assert bag_name in caller_locals, f"Failed to find {bag_name} in calling scope"

  if ckpt_prefix is None:
    ckpt_name = bag_name
  else:
    ckpt_name = f"{ckpt_prefix}_{bag_name}"

  caller_locals[bag_name] = checkpoint(
      name=ckpt_name,
      bag=caller_locals[bag_name],
      **kwargs,
  )

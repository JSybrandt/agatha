"""
This util is intended to be a universal initializer for all process-specific
helper data that is loaded at the start of the moliere construction process.
This is only intended for expensive complex structures that must be loaded at
startup, and we don't want to reload each function call.
"""

from typing import Callable, Any
from dask.distributed import Lock
# An initialize is a function that does not take arguments, and produces an
# expensive-to-load piece of data.

Initializer = Callable[[], Any]

_PROCESS_GLOBAL_DATA = {}

_INITIALIZERS = {}


def register(
    key:str,
    init:Callable,
)->None:
  """
  Adds the given initializer to the set.
  """
  assert key not in _INITIALIZERS
  _INITIALIZERS[key] = init


def _init(key:str)->None:
  assert key in _INITIALIZERS
  lock = Lock(f"init:{key}")
  while(not lock.acquire(timeout=5)):
    pass
  _PROCESS_GLOBAL_DATA[key] = _INITIALIZERS[key]()
  print("Initialized", key)
  lock.release()


def clear():
  "Deletes all of the process global data."
  _INITIALIZERS.clear()
  _PROCESS_GLOBAL_DATA.clear()


def get(key:str):
  if key not in _INITIALIZERS:
    raise Exception(f"Attempted to get unregistered key {key}")
  if key not in _PROCESS_GLOBAL_DATA:
    _init(key)
  # 2nd try
  if key not in _PROCESS_GLOBAL_DATA:
    raise Exception(f"Failed to initialize {key}")
  return _PROCESS_GLOBAL_DATA[key]

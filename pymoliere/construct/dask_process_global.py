"""
This util is intended to be a universal initializer for all process-specific
helper data that is loaded at the start of the moliere construction process.
This is only intended for expensive complex structures that must be loaded at
startup, and we don't want to reload each function call.
"""

from typing import Callable, Any
from dask.distributed import Lock, Worker, Client, get_worker
import cloudpickle
from pathlib import Path
# An initialize is a function that does not take arguments, and produces an
# expensive-to-load piece of data.

Initializer = Callable[[], Any]

_PROCESS_GLOBAL_DATA = {}

_INITIALIZERS = {}

PRELOADER_PATH = Path("~/.pymoliere_global_preloader").expanduser()

class WorkerPreloader(object):
  def __init__(self):
    self.initializers = {}

  def setup(self, worker:Worker):
    print("setup")
    worker._preloader_data = {}

  def teardown(self, worker:Worker):
    print("teardown")
    del worker._preloader_data

  def register(self, key:str, init:Callable)->None:
    "Adds a global object to the preloader"
    assert key not in self.initializers
    self.initializers[key] = init
    print("Registered", key)

  def get(self, key:str, worker:Worker)->Any:
    assert hasattr(worker, "_preloader_data")
    if key not in self.initializers:
      raise Exception(f"Attempted to get unregistered key {key}")
    if key not in worker._preloader_data:
      with worker._lock:
        if key not in worker._preloader_data:
          print(f"Initializing {key}")
          worker._preloader_data[key] = self.initializers[key]()
    return worker._preloader_data[key]

  def clear(self, worker):
    print("Clearing process global data.")
    del worker._preloader_data
    worker._preloader_data = {}


def add_global_preloader(client:Client, preloader:WorkerPreloader)->None:
  with open(PRELOADER_PATH, 'wb') as f:
    cloudpickle.dump(preloader, f)
  client.register_worker_plugin(preloader, name="global_preloader")


def get_global_preloader():
  worker = get_worker()
  if "global_preloader" not in worker.plugins:
    # This is run if the worker disconnects and reconnects.
    with open(PRELOADER_PATH, 'rb') as f:
      worker.plugins["global_preloader"] = cloudpickle.load(f)
      worker.plugins["global_preloader"].setup(worker)
  return worker.plugins["global_preloader"]


def get(key:str)->Any:
  "Gets a value from the global preloader"
  worker = get_worker()
  return get_global_preloader().get(key, worker)

def clear()->None:
  "Deletes all preloaded data. To be called following a ckpt."
  get_global_preloader().clear(get_worker())




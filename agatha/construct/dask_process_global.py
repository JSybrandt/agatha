"""
This util is intended to be a universal initializer for all process-specific
helper data that is loaded at the start of the construction process.
This is only intended for expensive complex structures that must be loaded at
startup, and we don't want to reload each function call.
"""

from typing import Callable, Any
from dask.distributed import Worker, Client, get_worker
import cloudpickle
from pathlib import Path
from multiprocessing import Lock
# An initialize is a function that does not take arguments, and produces an
# expensive-to-load piece of data.

Initializer = Callable[[], Any]
PRELOADER_PATH = Path("~/.agatha_global_preloader").expanduser()

class WorkerPreloader(object):
  def __init__(self):
    self.initializers = {}

  def setup(self, worker:Worker):
    worker._preloader_data = {}

  def teardown(self, worker:Worker):
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
    with worker._lock:
      del worker._preloader_data
      worker._preloader_data = {}

################################################################################

class LocalMockWorker(object):
  def __init__(self):
    "Because the LocalCluster won't spawn workers, we are going to mock one"
    self._lock = Lock()
    self.plugins = {}
    self._preloader_data = {}


LOCAL_MOCK_WORKER = LocalMockWorker()

def safe_get_worker():
  try:
    return get_worker()
  except:
    return LOCAL_MOCK_WORKER

def add_global_preloader(
    preloader:WorkerPreloader,
    client:Client=None,
)->None:
  with open(PRELOADER_PATH, 'wb') as f:
    cloudpickle.dump(preloader, f)
  if client is not None:
    client.register_worker_plugin(preloader, name="global_preloader")
  LOCAL_MOCK_WORKER.plugins["global_preloader"] = preloader
  LOCAL_MOCK_WORKER.plugins["global_preloader"].setup(LOCAL_MOCK_WORKER)


def get_global_preloader():
  worker = safe_get_worker()
  if "global_preloader" not in worker.plugins:
    # This is run if the worker disconnects and reconnects.
    with open(PRELOADER_PATH, 'rb') as f:
      worker.plugins["global_preloader"] = cloudpickle.load(f)
      worker.plugins["global_preloader"].setup(worker)
  return worker.plugins["global_preloader"]


def get(key:str)->Any:
  "Gets a value from the global preloader"
  return get_global_preloader().get(key, safe_get_worker())

def clear()->None:
  "Deletes all preloaded data. To be called following a ckpt."
  get_global_preloader().clear(safe_get_worker())

def get_worker_lock():
  return safe_get_worker()._lock

"""
This util is intended to be a universal initializer for all process-specific
helper data that is loaded at the start of the moliere construction process.
This is only intended for expensive complex structures that must be loaded at
startup, and we don't want to reload each function call.
"""

from typing import Callable, Any
from dask.distributed import Lock, Worker, Client, get_worker
# An initialize is a function that does not take arguments, and produces an
# expensive-to-load piece of data.

Initializer = Callable[[], Any]

_PROCESS_GLOBAL_DATA = {}

_INITIALIZERS = {}


class WorkerPreloader(object):
  def __init__(self):
    self.initializers = {}
    self.worker_data = {}

  def setup(self, worker:Worker):
    pass

  def teardown(self, worker:Worker):
    self.clear()

  def register(self, key:str, init:Callable)->None:
    "Adds a global object to the preloader"
    assert key not in self.initializers
    self.initializers[key] = init

  def initialize(self, key:str)->None:
    assert key in self.initializers
    lock = Lock(f"init:{key}")
    while(not lock.acquire(timeout=5)):
      pass
    self.worker_data[key] = self.initializers[key]()
    print("Initialized", key)
    lock.release()

  def clear(self)->None:
    self.worker_data.clear()
    self.initializers.clear()

  def get(self, key:str)->Any:
    if key not in self.initializers:
      raise Exception(f"Attempted to get unregistered key {key}")
    if key not in self.worker_data:
      self.initialize(key)
    # 2nd try
    if key not in self.worker_data:
      raise Exception(f"Failed to initialize {key}")
    return self.worker_data[key]

def add_global_preloader(client:Client, preloader:WorkerPreloader)->None:
  client.register_worker_plugin(preloader, name="global_preloader")

def get(key:str)->Any:
  "Gets a value from the global preloader"
  return get_worker().plugins["global_preloader"].get(key)

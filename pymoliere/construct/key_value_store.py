from pymoliere.construct import dask_process_global as dpg
from multiprocessing import Process
import socket
import asyncio
from distributed.core import rpc, Server
from typing import Any, Iterable, Tuple

def _start_kv_server(port:str):
  data = {}
  def put(comm, key, value):
    data[key] = value
  def get(comm, key):
    return data[key]
  server = Server({"put":put, "get":get})
  server.listen(f"tcp://:{port}")
  print(f"Launching local KV Store at: {socket.gethostname()}:{port}")
  asyncio.get_event_loop().run_forever()


def get_kv_server_initializer(port:str):
  """
  Starts the KV server in the background on whatever host calls this function.
  Returns an initializer that allows for other machines to contact the KV store.
  """
  # Start by launching the server in the background
  server_proc = Process(target=_start_kv_server, args=(port,))
  server_proc.start()
  server_host = socket.gethostname()
  def _init():
    # The init is only needed to tell workers where the server is located.
    return  f"tcp://{server_host}:{port}"
  return "kv_store:conn", _init


async def _put_many(conn:str, kv_pairs:Iterable[Tuple[Any, Any]])->None:
  with rpc(conn) as r:
    for k, v in kv_pairs:
      await r.put(key=k, value=v)


def put(key:Any, value:Any)->bool:
  """
  Sets the key-value pair in the local KV-Store. This requires
  get_kv_server_initializer to be added to dpg.
  """
  conn = dpg.get("kv_store:conn")
  asyncio.get_event_loop().run_until_complete(_put_many(conn, [(key, value)]))
  return True


def put_many(kv_pairs:Iterable[Tuple[Any, Any]])->Iterable[bool]:
  "Same as put, but writes all."
  conn = dpg.get("kv_store:conn")
  asyncio.get_event_loop().run_until_complete(_put_many(conn,kv_pairs))
  return [True]*len(kv_pairs)


async def _get_many(conn:str, keys:Iterable[Any])->Iterable[Any]:
  vals = []
  with rpc(conn) as r:
    for key in keys:
      vals.append(await r.get(key=key))
  return vals


def get(key)->Any:
  """
  Gets a previously stored kv pair from the local store.  This requires
  get_kv_server_initializer to be added to dpg.
  """
  conn = dpg.get("kv_store:conn")
  return (
      asyncio
      .get_event_loop()
      .run_until_complete(_get_many(conn,[key]))[0]
  )


def get_many(keys:Iterable[Any])->Iterable[Any]:
  "Same as get, but loads all."
  conn = dpg.get("kv_store:conn")
  return list(
      asyncio
      .get_event_loop()
      .run_until_complete(_get_many(conn, keys))
  )

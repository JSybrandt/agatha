import partd
from pymoliere.construct import dask_process_global as dpg
from typing import Iterable, Tuple, Any
from pathlib import Path
import socket
import random
from pymoliere.util.misc_util import iter_to_batches

def get_kv_server_initializer(
    server_file_path:Path,
    server_buffer_size:int,

)->Tuple[str, dpg.Initializer]:
  """
  Starts the KV server in the background on whatever host calls this function.
  Returns an initializer that allows for other machines to contact the KV store.
  """
  server = partd.Server(partd.Dict())

  # It is important that we copy the address. Otherwise when we reference
  # "server.address" in the init, we will require that the dpg coordination
  # capture the whole server value. Becuase we cannot serialize the server
  # using cloudpickle, that would cause an error. This way we only picle the
  # address.
  address = server.address.decode("utf-8")
  print(f"\t- Starting KV Store at: {address}")

  def _init():
    "Encodes data with Pickle, sends to server."

    # Client needs a serialization method The way this works is: first pickle
    # serializes the value, then bz2 compresses that, then we send the bytes
    # thru the client to the server.  When we read, we get a series of
    # compressed packets, each is unzipped, then unpickled, then concatenated
    # to our final output.  That last concatenation step requires that whatever
    # object we use implements "+"
    client = partd.Pickle(partd.BZ2(partd.Client(address)))

    # The test is to transmit an int and recover it
    key = random.randint(0, 10000)
    client.append({socket.gethostname(): [key]})
    if client.get(socket.gethostname())[-1] != key:
      raise Exception(f"{socket.gethostname()} failed to connect to {address}")
    return client
  return "kv_store:client", _init

def put_many(kv_pairs:Iterable[Tuple[Any,Any]])->Iterable[bool]:
  client = dpg.get("kv_store:client")
  for k, v in kv_pairs:
    client.append({k: v})
  # it is important that we follow the function signature of map_partitions for
  # better dask compatibility.
  return [True]

def get_many(keys:Iterable[Any])->Iterable[Any]:
  client = dpg.get("kv_store:client")
  print("Getting")
  res = []
  for k in keys:
    val = client.get(k)
    res += None if len(val) == 0 else val[-1]
  return res


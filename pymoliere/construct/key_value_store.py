import partd
from pymoliere.construct import dask_process_global as dpg
from typing import Iterable, Tuple, Any, Optional
from pathlib import Path
import socket
import random
from pymoliere.util.misc_util import iter_to_batches, Record
from pymoliere.util.partd_sqlite import SqliteInterface
import networkx as nx

def get_kv_server_initializer(
  persistent_server_path:Optional[Path]=None,
  client_name:str="client",
)->Tuple[str, dpg.Initializer]:
  """
  Starts the KV server in the background on whatever host calls this function.
  Returns an initializer that allows for other machines to contact the KV store.

  If persistent_server_path is specified, back the kv_store with sqlitedict.
  """
  if persistent_server_path is None:
    server = partd.Dict()
  else:
    server = SqliteInterface(persistent_server_path)
  server = partd.Server(server)

  # It is important that we copy the address. Otherwise when we reference
  # "server.address" in the init, we will require that the dpg coordination
  # capture the whole server value. Becuase we cannot serialize the server
  # using cloudpickle, that would cause an error. This way we only pickle the
  # address.
  address = server.address
  print(f"\t- Starting KV Store at: {address.decode('utf8')}")

  def _init():
    "Encodes data with Pickle, sends to server."

    # Client needs a serialization method The way this works is: first pickle
    # serializes the value, then bz2 compresses that, then we send the bytes
    # thru the client to the server.  When we read, we get a series of
    # compressed packets, each is unzipped, then unpickled, then concatenated
    # to our final output.  That last concatenation step requires that whatever
    # object we use implements "+"
    client = partd.Pickle(partd.BZ2(partd.Client(address)))

    # # The test is to transmit an int and recover it
    # key = random.randint(0, 10000)
    # client.append({socket.gethostname(): [key]})
    # if client.get(socket.gethostname())[-1] != key:
      # raise Exception(f"{socket.gethostname()} failed to connect to {address}")
    return client
  return f"kv_store:{client_name}", _init

def put_many(kv_pairs:Iterable[Tuple[Any,Any]], client_name:str="client")->None:
  client = dpg.get(f"kv_store:{client_name}")
  for b in iter_to_batches(kv_pairs, 64):
    with dpg.get_worker_lock():
      client.append({
        # Must append lists
        k: (v if isinstance(v, list) else [v]) for k, v in b
      })

def get_many(keys:Iterable[Any], client_name:str="client")->Iterable[Any]:
  client = dpg.get(f"kv_store:{client_name}")
  res = []
  for b in iter_to_batches(keys, 64):
    with dpg.get_worker_lock():
      res += [
          None if len(val) == 0 else val[-1]
          for val in client.get(b)
      ]
  return res


def write_records(
    records:Iterable[Record],
    client_name:str,
    id_field:str="id",
)->Iterable[int]:
  count = 0
  for record in records:
    kv_pairs = []
    field_names = []
    id_ = rec[id_field]
    for field_name, value in rec.items():
      if field_name == id_field:
        continue
      kv_pairs.append((f"{id_}.{field_name}", value))
      field_names.append(field_name)
    kv_pairs.append((id_,field_names))
    put_many(kv_pairs=kv_pairs, client_name=client_name)
    count += 1
  return [count]

def write_edges(
    subgraphs:Iterable[nx.Graph],
    client_name:str,
)->Iterable[int]:
  count = 0
  for subgraph in subgraphs:
    for source in subgraph.nodes:
      edges = [
        (target, attr["weight"]) for target, attr in subgraph[source].items()
      ]
      put_many(kv_pairs=[(source, edges)], client_name=client_name)
      count += 1
  return [count]

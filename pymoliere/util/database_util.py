import pymongo
from pymoliere.construct import dask_process_global as dpg
from typing import Union, List, Tuple, Iterable, Optional
from pymoliere.util.misc_util import Record
import dask
import dask.bag as dbag

# Object types

GENE_TYPE="a"
UMLS_TERM_TYPE="c"
DATA_BANK_TYPE="d"
ENTITY_TYPE="e"
LEMMA_TYPE="l"
MESH_TERM_TYPE="m"
NGRAM_TYPE="n"
PREDICATE_TYPE="p"
SENTENCE_TYPE="s"

# DPG Inti

# Inidices are values like -1 and "hashed"
MONGO_INDEX = Union[int, str]


def database_initializer(
    address:str,
    port:int,
    name:str,
)->Tuple[str, dpg.Initializer]:
  "Returns a pymongo database connection."
  def _init():
    client = pymongo.MongoClient(
        host=address,
        port=port)
    # This tests the connection
    client.server_info()
    return client[name]
  return "database:db", _init


def set_index(
    collection:str,
    field_name:str,
    index_type:MONGO_INDEX=pymongo.TEST,
)->None:
  db = dpg.get("database:db")
  db[collection].create_index([(field_name, index_type)])


def put(records:Iterable[Record], collection:str, **kwargs)->None:
  """
  Inserts all the records, returns the resulting count. Note that
  additional_args are allowed to specify delayed dependencies.
  """
  db = dpg.get("database:db")
  for r in records:
    try:
      db[collection].insert(r)
    except pymongo.errors.InvalidOperation as e:
      print("Encountered non-fatal issue:",  e)
      pass


def get(
    values:Iterable[str],
    collection:str,
    field_name:str,
    **kwargs
)->Iterable[Record]:
  db = dpg.get("database:db")
  return [db[collection].find_one({field_name: v}) for v in values]


def clear_collection(collection:str, **kwargs)->None:
  db = dpg.get("database:db")
  db[collection].drop()


def put_bag(
    bag:dbag.Bag,
    collection:str,
    indexed_field_name:Optional[str]=None,
    index_type:MONGO_INDEX=pymongo.TEXT,
)->dbag.Bag:
  """
  Writes all the records to collection. Sets index if specified. Returns a bag
  simply containing the number of written records, indented for use in the
  checkpointing system.
  """
  def put_part_wrapper(*args, **kwargs):
    put(*args, **kwargs)
    return [True]

  index_task = None
  if indexed_field_name is not None:
    print(f"\t- Setting index: {collection}.{field_name}:{index_type}")
    set_index_wrapper(
        collection=collection,
        field_name=indexed_field_name,
        index_type=index_type
    )
  return dbag.from_delayed([
    dask.delayed(put_part_wrapper)(
      records=part,
      collection=collection,
      index_task=index_task
    )
    for part in bag.to_delayed()
  ])

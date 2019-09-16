from itertools import zip_longest
from typing import List, Any
import numpy as np
import hashlib

def iter_to_batches(iterable, batch_size):
  args = [iter(iterable)] * batch_size
  for batch in zip_longest(*args):
    yield list(filter(lambda b: b is not None, batch))

def generator_to_list(*args, gen_fn=None, **kwargs):
  return [
      val for val in gen_fn(*args, **kwargs)
  ]

def flatten_list(list_of_lists:List[List[Any]])->List[Any]:
  return [item for sublist in list_of_lists for item in sublist]

def hash_str_to_int64(s):
  try:
    return np.int64(
        int.from_bytes(
          hashlib.md5(
            s.encode("utf-8")
          ).digest(),
          byteorder="big",
          signed=False,
        ) % np.iinfo(np.int64).max
    )
  except:
    raise(f"Error with: '{s}'")

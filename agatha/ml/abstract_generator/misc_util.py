from agatha.util.misc_util import hash_str_to_int
from typing import Any, Set, Iterable

class HashedIndex(object):
  """
  This class acts as a dict that maps items to a fixed range of values.
  Items must be convertible to strings.
  Value -> Idx
  idx -> Set of values
  """

  def __init__(self, max_index:int):
    self.max_index = max_index
    self.idx2elems = {}
    self.elem2idx = {}

  def add(self, elem:Any)->None:
    idx = hash_str_to_int(str(elem)) % self.max_index
    assert 0 <= idx < self.max_index
    if idx not in self.idx2elems:
      self.idx2elems[idx] = set()
    self.idx2elems[idx].add(elem)
    self.elem2idx[elem] = idx

  def get_index(self, elem:Any)->int:
    if elem in self.elem2idx:
      return self.elem2idx[elem]
    else:
      return None

  def get_elements(self, idx:int)->Set[Any]:
    assert 0 <= idx < self.max_index
    if idx in self.idx2elems:
      return self.idx2elems[idx]
    else:
      return None

  def __len__(self)->int:
    return self.max_index

  def has_element(self, elem:Any)->bool:
    return elem in self.elem2idx

  def has_index(self, idx:int)->bool:
    return idx in self.idx2elems

class OrderedIndex(object):
  """
  Same exact interface as hashed index, without the hashing
  """
  def __init__(self):
    self.max_index = None
    self.idx2elem = {}
    self.elem2idx = {}

  def add(self, elem:Any)->None:
    if self.max_index is None:
      idx = 0
      self.max_index = 1
    else:
      idx = self.max_index
      self.max_index += 1
    self.idx2elem[idx] = elem
    self.elem2idx[elem] = idx

  def get_index(self, elem:Any)->int:
    if elem in self.elem2idx:
      return self.elem2idx[elem]
    else:
      return None

  def get_elements(self, idx:int)->Set[Any]:
    assert 0 <= idx < self.max_index
    if idx in self.idx2elem:
      return {self.idx2elem[idx]}
    else:
      return None

  def __len__(self)->int:
    return self.max_index

  def has_element(self, elem:Any)->bool:
    return elem in self.elem2idx

  def has_index(self, idx:int)->bool:
    return idx in self.idx2elem

def items_to_hashed_index(collection:Iterable[Any], max_index:int)->HashedIndex:
  res = HashedIndex(max_index=max_index)
  for elem in collection:
    res.add(elem)
  return res

def items_to_ordered_index(collection:Iterable[Any])->OrderedIndex:
  res = OrderedIndex()
  for elem in collection:
    res.add(elem)
  return res

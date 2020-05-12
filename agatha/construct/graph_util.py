import dask.bag as dbag
from typing import Callable, Iterable, List
from agatha.util.misc_util import Record
from collections import defaultdict
import json

def record_to_bipartite_edges(
    records:dbag.Bag,
    get_neighbor_keys_fn:Callable[[Record], Iterable[str]],
    get_source_key_fn:Callable[[Record], str]=lambda x:x["id"],
    bidirectional:bool=True,
)->dbag.Bag:
  """
  This function is responsible for extracting edges from records. For example,
  if you had a bag of records, each containing a set of terms, you might want
  to get the set of edges between records and terms.

  Args:
    records: The collection of records we wish to extract edges from.
    get_neighbor_keys_fn: Given a record, return a list of graph keys that
      are adjacent to the given record
    get_source_key_fn: Given a record, return a graph key that uniquely
      identifies the root. By default we get the "id" field
    bidirectional: If true, we write record->neighbor and neighbor->record.
      If false, we only write record->neighbor.

  Returns:
    A bag containing serialized key-value pairs that can be used to create an
    Sqlite3LookupTable

  """

  def _to_kv(recs:Iterable[Record])->List[str]:
    "id, neighs to key_value strings"
    # Create graph, remove duplicate edges
    graph = defaultdict(set)
    for r in recs:
      id_ = r["id"]
      for neigh in r["neighs"]:
        graph[id_].add(neigh)
        if bidirectional:
          graph[neigh].add(id_)
    # Output edges
    res = []
    for source, targets in graph.items():
      for target in targets:
        res.append(json.dumps(dict(key=source, value=target)))
    return res

  return (
      records
      .map(lambda r: {
        "id": get_source_key_fn(r),
        "neighs": get_neighbor_keys_fn(r)
      })
      .map_partitions(_to_kv)
  )

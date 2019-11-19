import dask.bag as dbag
from typing import Callable, Set, Iterable, List
import networkx as nx
from pymoliere.util.misc_util import Record
from math import log
import pandas as pd

def nxgraphs_to_tsv_edge_list(graphs:Iterable[nx.Graph])->Iterable[str]:
  res = []
  for graph in graphs:
    for (source, target, data) in graph.edges(data=True):
      source = source.replace("\t", " ")
      target = target.replace("\t", " ")
      weight = data["weight"]
      res.append(f"{source}\t{target}\t{weight}")
  return res

def nxgraph_to_edge_records(graph:nx.Graph)->Iterable[Record]:
  return [
      {
        "source": s,
        "target": t,
        "weight": d["weight"],
      }
      for (s, t, d) in graph.edges(data=True)
  ]

def record_to_bipartite_edges(
    records:dbag.Bag,
    get_neighbor_keys_fn:Callable[[Record], List[str]],
    weight_by_tf_idf:bool=True,
    minimum_document_frequency:int=2,
    bidirectional:bool=True,
    default_weight_multiplier:float=1.0,
    get_source_key_fn:Callable[[Record], str]=lambda x:x["id"],
)->dbag.Bag:
  """
  This function is responsible for extracting edges from records. For example,
  if you had a bag of records, each containing a set of terms, you might want
  to get the set of edges between records and terms.

  @param records: The collection of records we wish to extract edges from.
  @param get_neighbor_keys_fn: Given a record, return a list of graph keys that
  are adjacent to the given record
  @param weight_by_tf_idf: If true, perform tf-idf weighting on edges. In this
  case, if t is a term, d is a document and C is a corpus, than we calculate
  1/((times t occurs in d / log(size of d)) * (size of C / number of d with t))
  @param minimum_document_frequency: only used if weight_by_tf_idf is true.
  Removes nodes among neighbors that don't occur frequently enough.
  @param bidirectional: If true, we write record->neighbor and neighbor->record.
  If false, we only write record->neighbor.
  @param default_weight_multiplier: All weights are multiplied by this. If we
  aren't calculating tf-idf, this is the value of every weight.
  @param get_source_key_fn: Given a record, return a graph key that uniquely
  identifies the root. By default we get the "id" field
  @return A collection of networkx subgraphs
  """

  def to_id_term_freq_len(records):
    res = []
    for record in records:
      id_ = get_source_key_fn(record)
      tfs = {}
      neighs = get_neighbor_keys_fn(record)
      for n in neighs:
        if n in tfs:
          tfs[n] += 1
        else:
          tfs[n] = 1
      res += [
          (id_, term, freq, len(neighs))
          for term, freq in tfs.items()
      ]
    # columns=id, term, freq, doc_len
    return res

  def to_partial_doc_freqs(records):
    t2df = {}
    for record in records:
      for t in set(get_neighbor_keys_fn(record)):
        if t in t2df:
          t2df[t] += 1
        else:
          t2df[t] = 1
    # columns=term, doc_freq
    return list(t2df.items())

  def calculate_tf_idf_part(part, corpus_size):
    res = []
    for row in part.itertuples():
      tfidf = 1.0 / ((
          row.freq / log(row.doc_len+1)
            ) * (
          log(float(corpus_size) / row.doc_freq)
      ))
      res.append([row.id, row.term, tfidf])
    return pd.DataFrame(res, columns=["id", "term", "freq"])

  def part_to_graph(id_term_freqs):
    graph = nx.Graph()
    for row in id_term_freqs:
      i, t, f = row[:3]
      f *= default_weight_multiplier
      graph.add_edge(i, t, weight=f)
      if bidirectional:
        graph.add_edge(t, i, weight=f)
    return [graph]

  # a list of (id, term, freq)
  term_df = (
      records
      .map_partitions(to_id_term_freq_len)
      .to_dataframe(
        meta={
          "id": str,
          "term": str,
          "freq": float,
          "doc_len": int,
        }
      )
  )
  if weight_by_tf_idf:
    # A list of (term, doc_freq)
    document_frequencies = (
        records
        .map_partitions(to_partial_doc_freqs)
        .to_dataframe(
          meta={
            "term": str,
            "doc_freq": int,
          }
        )
        .groupby("term")
        .sum()
    )
    # filter
    document_frequencies = document_frequencies[
        document_frequencies["doc_freq"] >= minimum_document_frequency
    ]

    corpus_size = records.count()
    term_df = (
        term_df
        .join(document_frequencies, how="inner", on="term")
        .map_partitions(
          calculate_tf_idf_part,
          corpus_size=corpus_size,
          meta={
            "id": str,
            "term": str,
            "freq": float,
          }
        )
    )

  return term_df.to_bag().map_partitions(part_to_graph)

# https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines
import dask.bag as dbag
import faiss
from pathlib import Path
import numpy as np
from typing import Iterable, List, Dict, Any, Callable, Optional, Tuple
from pymoliere.construct import (
    embedding_util,
    file_util,
    key_value_store as kv_store,
)
from pymoliere.util.misc_util import (
    iter_to_batches,
    hash_str_to_int64,
    flatten_list
)
from pymoliere.util import db_key_util
from pymoliere.util.misc_util import Record
import dask
import networkx as nx
from pymoliere.construct import dask_process_global as dpg
import pickle
from copy import copy


################################################################################

def get_faiss_index_initializer(
    faiss_index_path:Path,
    index_name:str="final"
)->Tuple[str, dpg.Initializer]:
  def _init():
    assert faiss_index_path.is_file()
    return faiss.read_index(str(faiss_index_path))
  return f"knn_util:faiss_{index_name}", _init


################################################################################

def write_inverted_index_to_kvstore(records:dbag.Bag)->dbag.core.Item:
  "Writes all hash-str pairs to local db, and returns count"
  def write_part_kv(records:Iterable[Record])->Iterable[bool]:
    vals = []
    for record in records:
      vals.append((
          hash_str_to_int64(record["id"]),
          db_key_util.to_graph_key(record["id"]),
      ))
    kv_store.put_many(vals)
    return [True]
  return records.map_partitions(write_part_kv).count()


def nearest_neighbors_network_from_index(
    records:dbag.Bag,
    hash_and_embedding:dbag.Bag,
    batch_size:int,
    num_neighbors:int,
    faiss_index_name="final",
    weight:float=1.0,
)->Iterable[nx.Graph]:
  """
  Applies faiss and runs results through inverted index. Requires
  knn_util:faiss_index and knn_util:inverted_index to be initialized.
  """
  def apply_faiss_to_edges(
      hash_and_embedding:Iterable[Record],
      total_written:int
  )->Iterable[nx.Graph]:

    # The only reason we need total_written is to make sure that the writing
    # happens before this point

    index = dpg.get(f"knn_util:faiss_{faiss_index_name}")
    inverted_index = {}

    graph = nx.Graph()
    for batch in iter_to_batches(hash_and_embedding, batch_size):
      hashes, embeddings = records_to_ids_and_embeddings(
          records=batch,
      )
      _, neighs_per_root = index.search(embeddings, num_neighbors)

      hashes = hashes.tolist() + flatten_list(neighs_per_root.tolist())
      hashes = list(set(hashes) - set(inverted_index.keys()))

      graph_keys = kv_store.get_many(hashes)
      for k, v in zip(hashes, graph_keys):
        inverted_index[k] = v

      # Create records
      for root_idx, neigh_indices in zip(hashes, neighs_per_root):
        root = inverted_index[root_idx]
        if root is None:
          continue
        for neigh_idx in neigh_indices:
          if neigh_idx == root_idx:
            continue
          neigh = inverted_index[neigh_idx]
          if neigh is None:
            continue
          graph.add_edge(root, neigh, weight=weight)
          graph.add_edge(neigh, root, weight=weight)
    return [graph]

  return (
      hash_and_embedding
      .map_partitions(
        apply_faiss_to_edges,
        write_inverted_index_to_kvstore(records),
      )
  )


def train_distributed_knn(
    hash_and_embedding:dbag.Bag,
    batch_size:int,
    num_centroids:int,
    num_probes:int,
    num_quantizers:int,
    bits_per_quantizer:int,
    training_sample_prob:float,
    shared_scratch_dir:Path,
    final_index_path:Path,
    id_field:str="id",
    embedding_field:str="embedding",
)->Path:
  """
  Computing all of the embeddings and then performing a KNN is a problem for memory.
  So, what we need to do instead is compute batches of embeddings, and use them in Faiss
  to reduce their dimensionality and process the appropriatly.

  I'm so sorry this one function has to do so much...

  @param hash_and_embedding: bag of hash value and embedding values
  @param text_field: input text field that we embed.
  @param id_field: output id field we use to store number ids
  @param batch_size: number of sentences per batch
  @param num_centroids: number of voronoi cells in approx nn
  @param num_probes: number of cells to consider when querying
  @param num_quantizers: number of sub-vectors to discritize
  @param bits_per_quantizer: bits per sub-vector
  @param shared_scratch_dir: location to store intermediate results.
  @param training_sample_prob: chance a point is trained on
  @return The path you can load the resulting FAISS index
  """
  init_index_path = shared_scratch_dir.joinpath("init.index")

  if not init_index_path.is_file():
    # First off, we need to get a representative sample for faiss training
    training_data = hash_and_embedding.random_sample(
        prob=training_sample_prob
    ).pluck(
        embedding_field
    )

    # Train initial index, store result in init_index_path
    init_index_path = dask.delayed(train_initial_index)(
      training_data=training_data,
      num_centroids=num_centroids,
      num_probes=num_probes,
      num_quantizers=num_quantizers,
      bits_per_quantizer=bits_per_quantizer,
      output_path=init_index_path,
    )

  # For each partition, load embeddings to idx
  partial_idx_paths = []
  for part_idx, part in enumerate(hash_and_embedding.to_delayed()):
    part_path=shared_scratch_dir.joinpath(f"part-{part_idx}.index")
    if part_path.is_file():  # rudimentary ckpt
      partial_idx_paths.append(dask.delayed(part_path))
    else:
      partial_idx_paths.append(
          dask.delayed(add_points_to_index)(
            records=part,
            init_index_path=init_index_path,
            output_path=part_path,
            batch_size=batch_size,
          )
      )

  return dask.delayed(merge_index)(
      init_index_path=init_index_path,
      partial_idx_paths=partial_idx_paths,
      final_index_path=final_index_path,
  )

def merge_index(
    init_index_path:Path,
    partial_idx_paths:List[Path],
    final_index_path:Path,
)->Path:
  init_index = faiss.read_index(str(init_index_path))
  for part_path in partial_idx_paths:
    part_idx = faiss.read_index(str(part_path))
    init_index.merge_from(part_idx, 0)
  faiss.write_index(init_index, str(final_index_path))
  return final_index_path


def train_initial_index(
    training_data:List[np.ndarray],
    num_centroids:int,
    num_probes:int,
    num_quantizers:int,
    bits_per_quantizer:int,
    output_path:Path,
)->Path:
  """
  Computes index using method from:
  https://hal.inria.fr/inria-00514462v2/document

  Vector dimensionality must be a multiple of num_quantizers.
  Input vectors are "chunked" into `num_quantizers` sub-components.
  Each chunk is reduced to a `bits_per_quantizer` value.
  Then, the L2 distances between these quantized bits are compared.

  For instance, a scibert embedding is 768-dimensional. If num_quantizers=32
  and bits_per_quantizer=8, then each vector is split into subcomponents of
  only 24 values, and these are further reduced to an 8-bit value. The result
  is that we're only using 1/3 of a bit per value in the input.

  When constructing the index, we use quantization along with the L2 metric to
  perform K-Means, constructing a voronoi diagram over our training data.  This
  allows us to partition the search space in order to make inference faster.
  `num_centroids` determines the number of voronoi cells a point could be in,
  while `num_probes` determines the number of nearby cells we consider at query
  time.  So higher centroids means faster and less accurate inference. Higher
  probes means the opposite, longer and more accurate queries.

  According to: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes,
  we should select #centroids on the order of sqrt(n).

  Choosing an index is hard:
  https://github.com/facebookresearch/faiss/wiki/Index-IO,-index-factory,-cloning-and-hyper-parameter-tuning
  """
  data = np.vstack([
    b.astype(dtype=np.float32) for b in training_data
  ])

  dim = data.shape[1]

  assert dim % num_quantizers == 0
  assert bits_per_quantizer in [8, 12, 16]

  coarse_quantizer = faiss.IndexFlatL2(dim)
  index = faiss.IndexIVFPQ(
      coarse_quantizer,
      dim,
      num_centroids,
      num_quantizers,
      bits_per_quantizer
  )
  index.nprobe = num_probes
  index.train(data)
  faiss.write_index(index, str(output_path))
  return output_path

def add_points_to_index(
    records:Iterable[Record],
    init_index_path:Path,
    batch_size:int,
    output_path:Path,
    embedding_field:str="embedding",
    id_field:str="id",
)->Path:
  "Loads an initial index, adds the partition to the index, and writes result"
  index = faiss.read_index(str(init_index_path))
  assert index.is_trained

  for batch in iter_to_batches(records, batch_size):
    ids, embeddings = records_to_ids_and_embeddings(
        records=batch,
        id_field=id_field,
        embedding_field=embedding_field
    )
    index.add_with_ids(embeddings, ids)
  faiss.write_index(index, str(output_path))
  return output_path

def records_to_ids_and_embeddings(
    records:Iterable[Record],
    id_field:str="id",
    embedding_field:str="embedding",
)->Tuple[np.ndarray, np.ndarray]:
  ids = np.array(
      list(map(lambda r:r[id_field], records)),
      dtype=np.int64
  )
  embeddings = np.array(
      list(map(lambda r:r[embedding_field], records)),
      dtype=np.float32
  )
  return ids, embeddings

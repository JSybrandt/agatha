# https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines
import pandas as pd
import dask.bag as dbag
import faiss
from pathlib import Path
import numpy as np
from typing import Iterable, List, Dict, Any, Callable, Optional, Tuple
from tqdm import tqdm
from pymoliere.construct import embedding_util
from pymoliere.util import file_util
from pymoliere.util.misc_util import (
    iter_to_batches,
    generator_to_list,
    flatten_list,
    hash_str_to_int64
)
from pymoliere.util import db_key_util
from pymoliere.util.misc_util import Record, Edge
import dask

# Maps a single hash value to a list of original id str
InvertedIds = Dict[int, List[str]]


def get_neighbors_from_index_per_part(
    records:Iterable[Record],
    inverted_ids: InvertedIds,
    num_neighbors:int,
    batch_size:int,
    index:Optional[faiss.Index]=None,
    index_path:Optional[Path]=None,
    id_field:str="id",
    embedding_field:str="embedding",
)->Iterable[Edge]:
  """
  Given a set of records, and a precomputed index object, actually get the KNN.
  Each record is embedded, and the given index is used to lookup similar
  records.  The inverted_ids object is also used to coordinate the numerical to
  string values.  Note that the result will NOT include a self-link, and may
  include more/less neighbors depending on hash collisions. (Effect should be
  negligible).
  """
  res = []
  self = get_neighbors_from_index_per_part
  if not hasattr(self, "index"):
    if index is not None:
      self.index = index
    elif index_path is not None:
      print("\t- Loading index from path...")
      self.index = faiss.read_index(str(index_path))
    assert hasattr(self, "index")
    assert self.index is not None

  for batch in iter_to_batches(records, batch_size):
    ids, embeddings = records_to_ids_and_embeddings(
        records=batch,
        id_field=id_field,
        embedding_field=embedding_field,
    )
    _, neigh_per_id = self.index.search(embeddings, num_neighbors)
    for root_idx, neigh_indices in zip(ids, neigh_per_id):
      # get text ids
      neigh_ids = flatten_list(
          inverted_ids[idx]
          for idx in neigh_indices
          if idx in inverted_ids and idx != root_idx
      )
      # Potential for hash collosions
      for root_key in inverted_ids[root_idx]:
        root_graph_key = db_key_util.to_graph_key(root_key)
        # for each text id
        for neigh_id in neigh_ids:
          neigh_graph_key = db_key_util.to_graph_key(neigh_id)
          res.append(db_key_util.to_edge(
            source=root_graph_key,
            target=neigh_graph_key
          ))
          res.append(db_key_util.to_edge(
            target=root_graph_key,
            source=neigh_graph_key
          ))
  return res

def create_inverted_index(
    ids:dbag.Bag,
)->InvertedIds:
  def part_to_inv_idx(
      part:Iterable[str],
  )->Iterable[InvertedIds]:
    res = {}
    for str_id in part:
      int_id = hash_str_to_int64(str_id)
      if int_id not in res:
        res[int_id] = [str_id]
      else:
        res[int_id].append(str_id)
    return [res]

  def merge_inv_idx(
      d1:InvertedIds,
      d2:InvertedIds=None,
  )->InvertedIds:
    if d2 is None:
      return d1
    if len(d2) > len(d1):
      d1, d2 = d2, d1
    for int_id, str_ids in d2.items():
      if int_id not in d1:
        d1[int_id] = str_ids
      else:
        d1[int_id] += str_ids
    return d1
  return ids.map_partitions(part_to_inv_idx).fold(merge_inv_idx)


def train_distributed_knn(
    idx_embedding:dbag.Bag,
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

  @param idx_embedding: bag of hash value and embedding values
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
  print("Cleaning up any tmp files...")
  for f in tqdm(shared_scratch_dir.iterdir()):
    if f.suffix == ".index" and f not in [init_index_path, final_index_path]:
      f.unlink()

  # First off, we need to get a representative sample for faiss training
  training_data = idx_embedding.random_sample(
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
  partial_idx_paths = idx_embedding.map_partitions(
      create_partial_faiss_index,
      # --
      init_index_path=init_index_path,
      shared_scratch_dir=shared_scratch_dir,
      batch_size=batch_size,
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

def create_partial_faiss_index(
    records:Iterable[Record],
    init_index_path:Path,
    shared_scratch_dir:Path,
    batch_size:int,
    embedding_field:str="embedding",
    id_field:str="id",
)->Path:
  "Loads an initial index, adds the partition to the index, and writes result"
  index = faiss.read_index(str(init_index_path))
  write_path = file_util.touch_random_unused_file(
      shared_scratch_dir,
      ".index"
  )
  assert index.is_trained

  for batch in iter_to_batches(records, batch_size):
    ids, embeddings = records_to_ids_and_embeddings(
        records=batch,
        id_field=id_field,
        embedding_field=embedding_field
    )
    index.add_with_ids(embeddings, ids)
  faiss.write_index(index, str(write_path))
  return write_path

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

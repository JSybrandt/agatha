# https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines
import pandas as pd
import dask.bag as dbag
import faiss
from pathlib import Path
import numpy as np
from typing import Iterable, List, Dict, Any, Callable, Optional
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

# Maps a single hash value to a list of original id str
InvertedIds = Dict[int, List[str]]


def get_neighbors_from_index_per_part(
    records:Iterable[Record],
    inverted_ids: InvertedIds,
    text_field:str,
    num_neighbors:int,
    scibert_data_dir:Path,
    batch_size:int,
    index:Optional[faiss.Index]=None,
    index_path:Optional[Path]=None,
    id_field:str="id"
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

  for rec_batch in iter_to_batches(records, batch_size):
    texts = [r[text_field] for r in rec_batch]
    ids = [r[id_field] for r in rec_batch]
    embs = next(embedding_util.embed_texts(
        texts=texts,
        scibert_data_dir=scibert_data_dir,
        batch_size=batch_size,
    ))
    if embs.shape[0] != len(texts):
      raise Exception("Error in:" + str(ids))
    _, neighbors = self.index.search(embs, num_neighbors)
    for root_id, neigh_indices in zip(ids, neighbors):
      root_idx = hash_str_to_int64(root_id)
      neigh_ids = flatten_list(
          inverted_ids[idx]
          for idx in neigh_indices
          if idx in inverted_ids and idx != root_idx
      )
      root_graph_key = db_key_util.to_graph_key(root_id)
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


def train_distributed_knn_from_text_fields(
    text_records:dbag.Bag,
    text_field:str,
    scibert_data_dir:Path,
    batch_size:int,
    num_centroids:int,
    num_probes:int,
    num_quantizers:int,
    bits_per_quantizer:int,
    training_sample_prob:float,
    shared_scratch_dir:Path,
    final_index_path:Path,
    id_field:str="id"
)->Path:
  """
  Computing all of the embeddings and then performing a KNN is a problem for memory.
  So, what we need to do instead is compute batches of embeddings, and use them in Faiss
  to reduce their dimensionality and process the appropriatly.

  I'm so sorry this one function has to do so much...

  @param text_records: bag of text dicts
  @param text_field: input text field that we embed.
  @param id_num_field: output id field we use to store number ids
  @param scibert_data_dir: location we can load scibert weights
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

  # Fileter for only those texts we care about, and index what we care about
  text_records = text_records.filter(
      lambda r: text_field in r and len(text_field) > 0
  ).map(lambda r:{
        "id_num": hash_str_to_int64(r[id_field]),
        "text": r[text_field],
  })

  if not init_index_path.is_file():
    # First off, we need to get a representative sample for faiss training
    print("\t- Getting representative sample:", training_sample_prob)
    training_data = text_records.random_sample(
        prob=training_sample_prob
    ).pluck(
        "text"
    ).map_partitions(
        generator_to_list,
        gen_fn=embedding_util.embed_texts,
        batch_size=batch_size,
        scibert_data_dir=scibert_data_dir,
    ).compute()

    training_data = np.vstack([
      b.astype(dtype=np.float32) for b in training_data
    ])

    print(training_data.shape)

    print(f"\t- Training on sample")
    init_index = train_initial_index(
      data=training_data,
      num_centroids=num_centroids,
      num_probes=num_probes,
      num_quantizers=num_quantizers,
      bits_per_quantizer=bits_per_quantizer,
    )
    print("\t- Saving index", init_index_path)
    faiss.write_index(init_index, str(init_index_path))
  else:
    print("\t- Using cached pretrained index!")
    init_index = faiss.read_index(str(init_index_path))

  print("\t- Adding points")
  partial_idx_paths = text_records.map_partitions(
      rw_embed_and_add_partition_to_idx,
      load_path=init_index_path,
      shared_scratch_dir=shared_scratch_dir,
      batch_size=batch_size,
      scibert_data_dir=scibert_data_dir,
  ).compute()

  print("\t- Merging points")
  for path in tqdm(partial_idx_paths):
    part_idx = faiss.read_index(str(path))
    init_index.merge_from(part_idx, 0)

  print("\t- Writing final index")
  faiss.write_index(init_index, str(final_index_path))

  return final_index_path

def train_initial_index(
    data:np.ndarray,
    num_centroids:int,
    num_probes:int,
    num_quantizers:int,
    bits_per_quantizer:int,
)->faiss.Index:
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
  print("\t\t- Training!!!")
  index.train(data)
  return index

def rw_embed_and_add_partition_to_idx(
    records:Iterable[Record],
    load_path:Path,
    shared_scratch_dir:Path,
    **kwargs,
)->List[Path]:
  partial_index_paths = []
  for record_batch in iter_to_batches(records, 300000):
    index = embed_and_add_partition_to_idx(
        records=record_batch,
        index=faiss.read_index(str(load_path)),
        **kwargs
    )
    print("Got:", index.ntotal)
    write_path = file_util.touch_random_unused_file(
        shared_scratch_dir,
        ".index"
    )
    faiss.write_index(index, str(write_path))
    partial_index_paths.append(write_path)
  return partial_index_paths

def embed_and_add_partition_to_idx(
    records:Iterable[Record],
    index:faiss.Index,
    scibert_data_dir:Path,
    batch_size:int,
    text_field:str="text",
    id_num_field:str="id_num",
)->faiss.Index:
  assert index.is_trained
  batches = embedding_util.embed_texts(
      texts=list(map(
        lambda r: r[text_field],
        records,
      )),
      scibert_data_dir=scibert_data_dir,
      batch_size=batch_size,
  )
  id_idx = 0
  for embeddings in batches:
    ids = np.array(
        list(map(
            lambda r: r[id_num_field],
            records[id_idx:id_idx+len(embeddings)],
        )),
        dtype=np.int64,
    )
    id_idx += len(embeddings)
    index.add_with_ids(embeddings, ids)
  return index

def add_partition_to_index(
    records:Iterable[Record],
    index:faiss.Index,
    embedding_field:str,
    id_num_field:str,
)->faiss.Index:
  "Output has to be a list for bag to join all"
  assert index.is_trained
  embeddings = np.array(
      list(map(lambda r:r[embedding_field], records)),
      dtype=np.float32
  )
  ids = np.array(
      list(map(lambda r:r[id_num_field], records)),
      dtype=np.int64
  )
  index.add_with_ids(embeddings, ids)
  return index

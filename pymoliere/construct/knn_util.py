# https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines
import pandas as pd
import dask.bag as dbag
import faiss
from pathlib import Path
import numpy as np
from typing import Iterable, List, Dict, Any, Callable, Optional
from pymoliere.construct import embedding_util
from pymoliere.util import file_util
from pymoliere.util.misc_util import iter_to_batches
from tqdm import tqdm

def train_distributed_knn_from_text_fields(
    text_records:dbag.Bag,
    id_fn:Callable[[Dict[str,Any]], str],
    text_field:str,
    scibert_data_dir:Path,
    batch_size:int,
    num_centroids:int,
    num_probes:int,
    num_quantizers:int,
    bits_per_quantizer:int,
    training_sample_prob:float,
    shared_scratch_dir:Path,
)->Path:
  """
  Computing all of the embeddings and then performing a KNN is a problem for memory.
  So, what we need to do instead is compute batches of embeddings, and use them in Faiss
  to reduce their dimensionality and process the appropriatly.

  I'm so sorry this one function has to do so much...


  @param text_records: bag of text dicts
  @param id_fn: fn that gets a string per dict, we hash this
  @param text_field: input text field that we embed.
  @param id_field: output id field we use to store string ids
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
  final_idx_path = shared_scratch_dir.joinpath("final.index")
  print("Cleaning up any tmp files...")
  for f in tqdm(shared_scratch_dir.iterdir()):
    if f.suffix == ".index" and f not in [init_index_path, final_idx_path]:
      f.unlink()

  # Fileter for only those texts we care about, and index what we care about
  text_records = text_records.filter(
      lambda r: text_field in r and len(text_field) > 0
  ).map(lambda r:{
        "id": hash(id_fn(r)),
        "text": r[text_field],
  })

  if not init_index_path.is_file():
    # First off, we need to get a representative sample for faiss training
    print("\t- Getting representative sample:", training_sample_prob)
    training_texts = text_records.random_sample(
        prob=training_sample_prob
    ).pluck(
        "text"
    ).compute()

    print(f"\t- Embedding on {len(training_texts)} values")
    batches = embedding_util.embed_texts(
        texts=training_texts,
        batch_size=batch_size,
        scibert_data_dir=scibert_data_dir,
        use_gpu=True,
    )
    training_data = None
    for training_batch in tqdm(batches):
      if training_data is None:
        training_data = training_batch.astype(dtype=np.float32)
      else:
        training_data = np.vstack((
          training_data,
          training_batch.astype(np.float32)
        ))

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
  faiss.write_index(init_index, str(final_idx_path))

  return final_idx_path

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
  index.train(data)
  return index

def rw_embed_and_add_partition_to_idx(
    records:Iterable[Dict[str, Any]],
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
    records:Iterable[Dict[str, Any]],
    index:faiss.Index,
    scibert_data_dir:Path,
    batch_size:int,
    text_field:str="text",
    id_num_field:str="id",
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
    records:Iterable[Dict[str, Any]],
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

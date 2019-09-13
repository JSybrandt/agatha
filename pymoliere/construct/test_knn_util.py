import dask.bag as dbag
from pymoliere.construct import knn_util
import numpy as np
import faiss
import dask
from pathlib import Path
import random
import faiss

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=1)
# client = Client(cluster)

def get_random_word():
  if not hasattr(get_random_word, "words"):
    with open("/usr/share/dict/words") as f:
      get_random_word.words = [l.strip().lower() for l in f]
  return random.choice(get_random_word.words)


def get_example_data(
    num_vectors:int,
    vec_dim:int,
    num_partitions:int=1,
)->dbag.Bag:
  return dbag.from_sequence([
    {"id": n, "embedding": np.random.rand(vec_dim)}
    for n in range(num_vectors)
    ],
    npartitions=num_partitions,
  )


def test_initial_index():
  data = np.array(
      get_example_data(
        num_vectors=500,
        vec_dim=32,
      ).pluck("embedding").compute(),
      dtype=np.float32,
  )
  init_index = knn_util.train_initial_index(
    data=data,
    num_centroids=4,
    num_probes=3,
    num_quantizers=8,
    bits_per_quantizer=8,
  )
  assert isinstance(init_index, faiss.Index)
  assert init_index.is_trained

def test_train_distributed_knn_from_text():
  num_docs = 5000
  text_records = dbag.from_sequence([{
        "text":get_random_word(),
        "id": str(i),
      }
      for i in range(num_docs)
    ],
    npartitions=5,
  )

  actual = knn_util.train_distributed_knn_from_text_fields(
    text_records=text_records,
    id_fn=lambda r: r["id"],
    text_field="text",
    scibert_data_dir=Path("data/scibert_scivocab_uncased"),
    batch_size=4096,
    num_centroids=4,
    num_probes=1,
    num_quantizers=32,
    bits_per_quantizer=8,
    training_sample_prob=1,
    shared_scratch_dir=Path("/tmp"),
  )
  assert actual.is_file()
  actual_idx = faiss.read_index(str(actual))
  assert actual_idx.ntotal == num_docs


def test_inverted_index():
  num_docs = 100
  input_objs = [{
      "text":get_random_word(),
      "id": str(i),
    }
    for i in range(num_docs)
  ]
  expected = {
      hash(e["id"]): [e["id"]]
      for e in input_objs
  }
  input_ids = dbag.from_sequence(
    input_objs,
    npartitions=5,
  ).map(
    lambda x: x["id"]
  )
  actual = knn_util.create_inverted_index(
    ids=input_ids,
  ).compute()
  assert actual == expected


# def test_add_points_to_index():
  # records=get_example_data(
    # num_vectors=500,
    # vec_dim=32,
  # )
  # init_index = knn_util.train_initial_index(
    # records=records.pluck("embedding").compute(),
    # num_centroids=4,
    # num_probes=3,
    # num_quantizers=8,
    # bits_per_quantizer=8,
  # )
  # index_with_vec = knn_util.add_partition_to_index(
      # records=records.compute(),
      # index=init_index,
      # embedding_field="embedding",
      # id_num_field="id",
  # )
  # assert isinstance(index_with_vec, faiss.Index)
  # assert index_with_vec.is_trained
  # assert index_with_vec.ntotal == 500

# def test_distributed_train_index():
  # records = get_example_data(
      # num_vectors=2000,
      # vec_dim=32,
      # num_partitions=4,
  # )
  # training_subset = records.random_sample(0.2).pluck("embedding").compute()
  # print("Ping", 1)
  # init_index = knn_util.train_initial_index(
    # training_subset,
    # num_centroids=4,
    # num_probes=3,
    # num_quantizers=8,
    # bits_per_quantizer=8,
  # )
  # print("Ping", 2)
  # indices = records.map_partitions(
      # knn_util.add_partition_to_index,
      # index=init_index,
      # embedding_field="embedding",
      # id_num_field="id",
  # )
  # print("Ping", 3)
  # real_indices = indices.compute()
  # print("Ping", 4)
  # assert len(real_indices) == 4
  # for idx in real_indices[1:]:
    # real_indices[0].merge_from(idx, real_indices[0].d)
  # assert real_indices[0].ntotal == 2000

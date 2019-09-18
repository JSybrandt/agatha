from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
    dask_config,
)
from pymoliere.util import file_util, ftp_util
from pymoliere.construct import (
    parse_pubmed_xml,
    text_util,
    knn_util,
    embedding_util,
)
from dask.distributed import Client, LocalCluster, as_completed
import dask.bag as dbag
import dask
from pathlib import Path
import faiss
from copy import copy
from typing import Dict, Any, Callable
from random import shuffle
import redis
import socket
from pymoliere.construct.write_db import(
    init_redis_client,
    write_record,
    write_edge,
)
from pymoliere.util.misc_util import hash_str_to_int64


if __name__ == "__main__":
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere build with the following custom parameters:")
  print(config)

  # Checks
  print("Performing config checks")
  shared_scratch_root = Path(config.cluster.shared_scratch)
  shared_scratch_root.mkdir(parents=True, exist_ok=True)
  assert shared_scratch_root.is_dir()
  local_scratch_root = Path(config.cluster.local_scratch)
  local_scratch_root.mkdir(parents=True, exist_ok=True)
  assert local_scratch_root.is_dir()

  def mk_scratch(task_name):
    "returns local, shared"
    return file_util.prep_scratches(
      local_scratch_root=local_scratch_root,
      shared_scratch_root=shared_scratch_root,
      task_name=task_name,
    )

  # Configure Dask
  dask_config.set_local_tmp(local_scratch_root)
  dask_config.set_verbose_worker()

  if config.cluster.run_locally:
    print("Running on local machine!")
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    dask_client = Client(cluster)
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)

  if config.cluster.restart:
    print("\t- Restarting cluster...")
    dask_client.restart()
    datasets = dask_client.list_datasets()
    for d in datasets:
      dask_client.unpublish_dataset(d)

  print(f"\t- Running on {len(dask_client.nthreads())} machines.")

  print("Connecting to Redis...")
  redis_client = redis.Redis(
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  )
  if config.db.clear:
    print("\t- Wiping existing DB")
    redis_client.flushdb()

  if config.db.address == "localhost":
    config.db.address = socket.gethostname()
    print(f"\t- Renaming localhost to {config.db.address}")

  # Initialize Helper Objects
  print("Initializing Helper Objects")
  print(f"\t- scispacy NLP parser")
  dask_client.run(
      text_util.init_analyze_sentence,
      # --
      scispacy_version=config.parser.scispacy_version,
  )
  print(f"\t- scibert transformer")
  dask_client.run(
      embedding_util.init_model,
      # --
      scibert_data_dir=config.parser.scibert_data_dir,
      disable_gpu=config.sys.disable_gpu,
  )
  print(f"\t- redis client")
  dask_client.run(
      init_redis_client,
      # --
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  )

  # Prepping all scratch dirs
  print("Prepping scratch directories")
  download_local, download_shared = mk_scratch("download_pubmed")
  _, faiss_index_dir = mk_scratch("faiss_index")

  # Download all of pubmed.
  print("Downloading pubmed XML Files")
  with ftp_util.ftp_connect(
      address=config.ftp.address,
      workdir=config.ftp.workdir,
  ) as conn:
    # Downloads new files if not already present in shared
    xml_paths = ftp_util.ftp_retreive_all(
        conn=conn,
        pattern="^.*\.xml\.gz$",
        directory=download_shared,
        show_progress=True,
    )

  # READY TO GO!
  print("Preparing computation graph")
  if config.debug.enable:
    print(f"\t- Downsampling {len(xml_paths)} xml files to only "
          f"{config.debug.partition_subset_size}.")
    shuffle(xml_paths)
    xml_paths = xml_paths[:config.debug.partition_subset_size]

  # Parse xml-files per-partition
  pubmed_documents = dbag.from_delayed([
    dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
      xml_path=p,
      local_scratch=download_local
    )
    for p in xml_paths
  ]).filter(
    # Only take the english ones
    lambda r: r["language"]=="eng"
  )

  if config.debug.enable:
    print("\t- Downsampling documents by"
          f"{config.debug.document_sample_rate}")
    pubmed_documents = pubmed_documents.random_sample(
        config.debug.document_sample_rate,
    )

  # Split documents into sentences, filter out too-long and too-short sentences.
  pubmed_sentences = pubmed_documents.map(
      text_util.split_sentences,
      min_sentence_len=config.parser.min_sentence_len,
      max_sentence_len=config.parser.max_sentence_len,
  ).flatten()

  # Add POS tagging, lemmas, entitites, and additional data to each sent
  pubmed_sent_w_ent = pubmed_sentences.map(
      text_util.analyze_sentence,
      # --
      text_field="sent_text",
  )
  # Store result in redis db
  write_sent_with_ent = pubmed_sent_w_ent.map(write_record)

  # get edges from sentence metadata and store in DB
  write_sentence_meta_edges = pubmed_sent_w_ent.map_partitions(
      text_util.get_edges_from_sentence_part
  ).map(
      write_edge
  )

  # Get KNN. To start, we need numeric indices and embeddings
  idx_embedding = pubmed_sentences.map(
      lambda x: {
        "id": hash_str_to_int64(x["id"]),  # need numeric ids
        "sent_text": x["sent_text"],  # keep text
      }
  ).map_partitions(
      embedding_util.embed_records,
      # --
      batch_size=config.sys.batch_size,
      text_field="sent_text",
  )

  # need map from numeric id to hash val
  inverted_ids = knn_util.create_inverted_index(
      ids=pubmed_sent_w_ent.pluck("id")
  )

  # Now we can distribute the knn training
  final_index_path = knn_util.train_distributed_knn(
      idx_embedding=idx_embedding,
      batch_size=config.sys.batch_size,
      num_centroids=config.sentence_knn.num_centroids,
      num_probes=config.sentence_knn.num_probes,
      num_quantizers=config.sentence_knn.num_quantizers,
      bits_per_quantizer=config.sentence_knn.bits_per_quantizer,
      training_sample_prob=config.sentence_knn.training_probability,
      shared_scratch_dir=faiss_index_dir,
      final_index_path=faiss_index_dir.joinpath("final.index"),
  )

  # Finally, we can compute NN for each element
  write_nearest_neighbors_edges = idx_embedding.map_partitions(
      knn_util.get_neighbors_from_index_per_part,
      # --
      inverted_ids=inverted_ids,
      num_neighbors=config.sentence_knn.num_neighbors,
      batch_size=config.sys.batch_size,
      index_path=final_index_path,
  ).map(
      write_edge
  )

  print("Running!!!")
  dask_client.compute(
    [
      write_nearest_neighbors_edges,
      write_sentence_meta_edges,
      write_sent_with_ent,
    ],
    sync=True,
  )

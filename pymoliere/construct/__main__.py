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
from random import random
import redis
import socket
from pymoliere.construct.write_db import(
    init_redis_client,
    write_record,
    write_edge,
)


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

  if config.cluster.run_locally:
    print("Running on local machine!")
    cluster = LocalCluster(n_workers=1, threads_per_worker=1)
    dask_client = Client(cluster)
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t{cluster_address}")
    dask_client = Client(address=cluster_address)

  if config.cluster.restart:
    print("Restarting cluster...")
    dask_client.restart()
    datasets = dask_client.list_datasets()
    for d in datasets:
      dask_client.unpublish_dataset(d)

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

  # READY TO GO!
  _, out_sent_w_ent = mk_scratch("pubmed_sent_w_ent")
  if not file_util.is_result_saved(out_sent_w_ent):
    print("Checking / Retrieving PubMed")
    download_local, download_shared = file_util.prep_scratches(
      local_scratch_root=local_scratch_root,
      shared_scratch_root=shared_scratch_root,
      task_name="download_pubmed",
    )
    with ftp_util.ftp_connect(
        address=config.ftp.address,
        workdir=config.ftp.workdir,
    ) as conn:
      xml_paths = ftp_util.ftp_retreive_all(
          conn=conn,
          pattern="^.*\.xml\.gz$",
          directory=download_shared,
          show_progress=True,
      )

    if config.HasField("debug_sample_rate"):
      xml_paths = [x for x in xml_paths if random() < config.debug_sample_rate]
      print(f"\t- Downsampled XML files to {len(xml_paths)} files.")

    print("Splitting sentences.")
    pubmed_sentences = dbag.from_delayed([
      dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
        xml_path=p,
        local_scratch=download_local
      )
      for p in xml_paths
    ]).filter(lambda r: r["language"]=="eng"
    ).map(
        text_util.split_sentences,
        min_sentence_len=config.parser.min_sentence_len,
        max_sentence_len=config.parser.max_sentence_len,
    ).flatten()

    if config.HasField("debug_sample_rate"):
      pubmed_sentences = pubmed_sentences.random_sample(
          config.debug_sample_rate
      )
      print(f"\t- Downsampled pubmed sentences by {config.debug_sample_rate}")

    print("Analyzing each document.")
    print("\t- Initializing helper object")
    dask_client.run(
        text_util.init_analyze_sentence,
        # --
        scispacy_version=config.parser.scispacy_version,
    )
    print("\t- Done!")
    pubmed_sent_w_ent = pubmed_sentences.map(
        text_util.analyze_sentence,
        # --
        text_field="sent_text",
    )
    print("\t- Saving...")
    file_util.save(pubmed_sent_w_ent, out_sent_w_ent)

  # Once we have our initial processing complete, its important that we always
  # retrieve these results from storage. Otherwise, we may compute again.
  # Furthermore, we can't actually afford to keep this whole thing persisted.
  pubmed_sent_w_ent = file_util.load(out_sent_w_ent)

  _, tmp_faiss_index_dir = mk_scratch("tmp_faiss_index")
  final_index_path = tmp_faiss_index_dir.joinpath("final.index")
  if not final_index_path.is_file():
    print("Training KNN for sentence embeddings.")
    knn_util.train_distributed_knn_from_text_fields(
        text_records=pubmed_sent_w_ent,
        text_field="sent_text",
        scibert_data_dir=config.parser.scibert_data_dir,
        batch_size=config.parser.batch_size,
        num_centroids=config.sentence_knn.num_centroids,
        num_probes=config.sentence_knn.num_probes,
        num_quantizers=config.sentence_knn.num_quantizers,
        bits_per_quantizer=config.sentence_knn.bits_per_quantizer,
        training_sample_prob=config.sentence_knn.training_probability,
        shared_scratch_dir=tmp_faiss_index_dir,
        final_index_path=final_index_path,
    )

  # Get a map from ID to documents
  inverted_ids = knn_util.create_inverted_index(
      ids=pubmed_sent_w_ent.pluck("id")
  )
  # Get NN Edges
  print("Getting Nearest-Neighbors per-sentence")
  # Bag of Edge
  nearest_neighbors_edges = pubmed_sent_w_ent.map_partitions(
      knn_util.get_neighbors_from_index_per_part,
      # --
      inverted_ids=inverted_ids,
      text_field="sent_text",
      num_neighbors=config.sentence_knn.num_neighbors,
      scibert_data_dir=config.parser.scibert_data_dir,
      batch_size=config.parser.batch_size,
      index_path=final_index_path,
  )

  print("Getting edges per-sentence")
  # Bag of Edge
  sentence_meta_edges = pubmed_sent_w_ent.map_partitions(
      text_util.get_edges_from_sentence_part
  )

  print("Initializing redis connection per-worker")
  dask_client.run(
      init_redis_client,
      # --
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  )
  write_all_edges = dbag.concat([
    nearest_neighbors_edges,
    sentence_meta_edges,
  ]).map(write_edge)
  write_all_sents = pubmed_sent_w_ent.map(write_record)
  print("Running")
  dask_client.compute(write_all_edges, write_all_sents)

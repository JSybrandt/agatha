    text_util,
    write_db,
)
from pymoliere.util import misc_util
from dask.distributed import (
  Client,
  LocalCluster,
)
from copy import copy
from pathlib import Path
from random import shuffle
from typing import Dict, Any, Callable
import dask
import dask.bag as dbag
import faiss
import redis
import socket
from pprint import pprint


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

  # Connect
  if config.cluster.run_locally:
    print("Running on local machine!")
    cluster = LocalCluster()
    dask_client = Client(cluster)
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address, heartbeat_interval=500)
  if config.cluster.restart:
    print("\t- Restarting cluster...")
    dask_client.restart()
  print(f"\t- Running on {len(dask_client.nthreads())} machines.")

  # Configure Redis ##############
  print("Connecting to Redis...")
  redis_client = redis.Redis(
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  )
  try:
    redis_client.ping()
  except:
    raise Exception(f"No redis server running at {config.db.address}")
  if config.db.clear:
    print("\t- Wiping existing DB")
    redis_client.flushdb()
  if config.db.address == "localhost":
    config.db.address = socket.gethostname()
    print(f"\t- Renaming localhost to {config.db.address}")

  # Versioning info
  meta_data = write_db.get_meta_record(config)
  write_db.write_record(
      rec=meta_data,
      redis_client=redis_client
  )
  print("Metadata:")
  pprint(meta_data)

  # Initialize Helper Objects ###
  print("Registering Helper Objects")
  def prepare_dask_process_global():
    dpg.clear()
    dpg.register(*text_util.get_scispacy_initalizer(
        scispacy_version=config.parser.scispacy_version,
    ))
    dpg.register(*text_util.get_stopwordlist_initializer(
        stopword_path=config.parser.stopword_list
    ))
    dpg.register(*embedding_util.get_scibert_initializer(
        scibert_data_dir=config.parser.scibert_data_dir,
        disable_gpu=config.sys.disable_gpu,
    ))
    dpg.register(*write_db.get_redis_client_initialzizer(
        host=config.db.address,
        port=config.db.port,
        db=config.db.db_num,
    ))
  dask_client.run(prepare_dask_process_global)

  # Prepping all scratch dirs ###
  print("Prepping scratch directories")
  download_local, download_shared = mk_scratch("download_pubmed")
  _, faiss_index_dir = mk_scratch("faiss_index")
  _, checkpoint_dir = mk_scratch("dask_checkpoints")

  # Download all of pubmed. ####
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
  # an attempt to keep partitions the same each call
  xml_paths.sort()

  ##############################################################################

  # READY TO GO!
  print("Preparing computation graph")
  if config.debug.enable:
    print(f"\t- Downsampling {len(xml_paths)} xml files to only "
          f"{config.debug.partition_subset_size}.")
    # Takes the top x (typically larger)
    xml_paths = xml_paths[-config.debug.partition_subset_size:]

  # Parse xml-files per-partition
  pubmed_documents = dbag.from_delayed([
    dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
      xml_path=p,
    )
    for p in xml_paths
  ]).filter(
    # Only take the english ones
    lambda r: r["language"]=="eng"
  )

  if config.debug.enable:
    print("\t- Downsampling documents by "
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
  pubmed_sent_w_ent = pubmed_sentences.map_partitions(
      text_util.analyze_sentences,
      # --
      text_field="sent_text",
  )

  pubmed_sent_w_ent = dask_checkpoint.checkpoint(
      pubmed_sent_w_ent,
      name="pubmed_sent_w_ent",
      checkpoint_dir=checkpoint_dir,
  )

  # Perform n-gram mining, introduces a new field "ngrams"
  pubmed_sent_w_ngrams = text_util.get_frequent_ngrams(
      analyzed_sentences=pubmed_sent_w_ent,
      max_ngram_length=config.phrases.max_ngram_length,
      min_ngram_support=config.phrases.min_ngram_support,
      min_ngram_support_per_partition=\
          config.phrases.min_ngram_support_per_partition,
  )

  pubmed_sent_w_ngrams = dask_checkpoint.checkpoint(
      pubmed_sent_w_ngrams,
      name="pubmed_sent_w_ngrams",
      checkpoint_dir=checkpoint_dir,
  )

  # Store result in redis db
  write_sent_with_ent = pubmed_sent_w_ngrams.map(
      text_util.add_bow_to_analyzed_sentence
  ).map(
      write_db.write_record
  )

  # get edges from sentence metadata and store in DB
  write_sentence_meta_edges = pubmed_sent_w_ngrams.map_partitions(
      text_util.get_edges_from_sentence_part,
      # --
      document_freqs=text_util.get_document_frequencies(pubmed_sent_w_ngrams),
      total_documents=pubmed_sent_w_ngrams.count(),
  ).map(
      write_db.write_edge
  )

  # Get KNN. To start, we need numeric indices and embeddings
  idx_embedding = pubmed_sentences.map(
      lambda x: {
        "id": misc_util.hash_str_to_int64(x["id"]),  # need numeric ids
        "sent_text": x["sent_text"],  # keep text
      }
  ).map_partitions(
      embedding_util.embed_records,
      # --
      batch_size=config.sys.batch_size,
      text_field="sent_text",
      max_sequence_length=config.parser.max_sequence_length,
  )

  idx_embedding = dask_checkpoint.checkpoint(
      idx_embedding,
      name="idx_embedding",
      checkpoint_dir=checkpoint_dir,
  )

  # need map from numeric id to hash val
  inverted_ids = knn_util.create_inverted_index(
      ids=pubmed_sent_w_ngrams.pluck("id")
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
      write_db.write_edge
  )

  print("Running!!!")
  tasks = [
      write_nearest_neighbors_edges,
      write_sentence_meta_edges,
      write_sent_with_ent,
  ] + dask_checkpoint.get_checkpoint_tasks()
  tasks = dask.optimize(tasks)
  dask_client.compute(tasks, sync=True)

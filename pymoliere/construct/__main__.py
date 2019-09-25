from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import (
    dask_process_global as dpg,
    dask_checkpoint,
    embedding_util,
    file_util,
    ftp_util,
    knn_util,
    parse_pubmed_xml,
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
import shutil


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
  write_db.write_records(
      [meta_data],
      redis_client=redis_client
  )

  # Initialize Helper Objects ###
  print("Registering Helper Objects")
  preloader = dpg.WorkerPreloader()
  preloader.register(*text_util.get_scispacy_initalizer(
      scispacy_version=config.parser.scispacy_version,
  ))
  preloader.register(*text_util.get_stopwordlist_initializer(
      stopword_path=config.parser.stopword_list
  ))
  preloader.register(*embedding_util.get_scibert_initializer(
      scibert_data_dir=config.parser.scibert_data_dir,
      disable_gpu=config.sys.disable_gpu,
  ))
  preloader.register(*write_db.get_redis_client_initialzizer(
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  ))
  dpg.add_global_preloader(client=dask_client, preloader=preloader)

  # Prepping all scratch dirs ###
  def scratch(task_name):
    "Creates a local / global scratch dir with the give name"
    return file_util.prep_scratches(
      local_scratch_root=local_scratch_root,
      shared_scratch_root=shared_scratch_root,
      task_name=task_name,
    )

  print("Prepping scratch directories")
  download_local, download_shared = scratch("download_pubmed")
  _, faiss_index_dir = scratch("faiss_index")
  _, checkpoint_dir = scratch("dask_checkpoints")

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

  if config.clear_checkpoints:
    print("Clearing checkpoint dir")
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()

  def ckpt(name:str)->None:
    "Applies checkpointing to the given bag"
    assert name in globals()
    assert type(globals()[name]) == dbag.Bag
    globals()[name] = dask_checkpoint.checkpoint(
        globals()[name],
        name=name,
        checkpoint_dir=checkpoint_dir,
    )

  ##############################################################################

  print("Preparing computation graph")
  final_tasks = []
  if config.debug.enable:
    print(f"\t- Downsampling {len(xml_paths)} xml files to only "
          f"{config.debug.partition_subset_size}.")
    # Takes the top x (typically larger)
    xml_paths = xml_paths[-config.debug.partition_subset_size:]

  # Parse xml-files per-partition
  medline_documents = dbag.from_delayed([
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
    medline_documents = medline_documents.random_sample(
        config.debug.document_sample_rate,
    )
  ckpt("medline_documents")

  # Split documents into sentences, filter out too-long and too-short sentences.
  sentences = medline_documents.map_partitions(
      text_util.split_sentences,
      # --
      min_sentence_len=config.parser.min_sentence_len,
      max_sentence_len=config.parser.max_sentence_len,
  )
  ckpt("sentences")

  # Add POS tagging, lemmas, entitites, and additional data to each sent
  sentences_with_lemmas = sentences.map_partitions(
      text_util.analyze_sentences,
      # --
      text_field="sent_text",
  )
  ckpt("sentences_with_lemmas")

  # Perform n-gram mining, introduces a new field "ngrams"
  sentences_with_ngrams = text_util.get_frequent_ngrams(
      analyzed_sentences=sentences_with_lemmas,
      max_ngram_length=config.phrases.max_ngram_length,
      min_ngram_support=config.phrases.min_ngram_support,
      min_ngram_support_per_partition=\
          config.phrases.min_ngram_support_per_partition,
  )
  ckpt("sentences_with_ngrams")

  sentences_with_bow = sentences_with_ngrams.map_partitions(
      text_util.add_bow_to_analyzed_sentence
  )
  ckpt("sentences_with_bow")

  final_tasks.append(sentences_with_bow.map_partitions(write_db.write_records))

  # get edges from sentence metadata and store in DB
  sentence_edges = sentences_with_bow.map_partitions(
      text_util.get_edges_from_sentence_part,
      # --
      document_freqs=text_util.get_document_frequencies(sentences_with_bow),
      total_documents=sentences_with_bow.count(),
  )
  ckpt("sentence_edges")

  final_tasks.append(sentence_edges.map_partitions(write_db.write_edges))

  # At this point we have to do the embedding

  # Get KNN. To start, we need numeric indices and embeddings. Additionally,
  # we're going to need the opposite mapping as well.
  strid2hash= knn_util.create_inverted_index(sentences)
  hash_and_embedding = sentences.map(
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
  ckpt("hash_and_embedding")

  # Now we can distribute the knn training
  final_index_path = knn_util.train_distributed_knn(
      idx_embedding=hash_and_embedding,
      batch_size=config.sys.batch_size,
      num_centroids=config.sentence_knn.num_centroids,
      num_probes=config.sentence_knn.num_probes,
      num_quantizers=config.sentence_knn.num_quantizers,
      bits_per_quantizer=config.sentence_knn.bits_per_quantizer,
      training_sample_prob=config.sentence_knn.training_probability,
      shared_scratch_dir=faiss_index_dir,
      final_index_path=faiss_index_dir.joinpath("final.index"),
  )

  nearest_neighbors_edges = hash_and_embedding.map_partitions(
      knn_util.get_neighbors_from_index_per_part,
      # --
      inverted_ids=strid2hash,
      num_neighbors=config.sentence_knn.num_neighbors,
      batch_size=config.sys.batch_size,
      index_path=final_index_path,
  )
  ckpt("nearest_neighbors_edges")

  final_tasks.append(
      nearest_neighbors_edges.map_partitions(write_db.write_edges)
  )

  print("Running!")
  final_tasks += dask_checkpoint.get_checkpoint_tasks()
  final_tasks = dask.optimize(final_tasks)
  dask_client.compute(final_tasks, sync=True)

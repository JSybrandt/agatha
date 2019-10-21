from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import (
    dask_checkpoint,
    dask_process_global as dpg,
    embedding_util,
    file_util,
    ftp_util,
    graph_util,
    knn_util,
    key_value_store,
    parse_pubmed_xml,
    text_util,
    write_db,
)
from pymoliere.util import misc_util
from dask.distributed import (
  Client,
)
from pymoliere.ml.sentence_classifier import (
    SentenceClassifier,
    LABEL2IDX as SENT_TYPE_SET,
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
from pymoliere.util.db_key_util import to_graph_key


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
    # Changes to dpg allow for a "none" dask client
    dask_client = None
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
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
  faiss_index_path = faiss_index_dir.joinpath("final.index")

  # Initialize Helper Objects ###
  print("Registering Helper Objects")
  preloader = dpg.WorkerPreloader()
  preloader.register(*key_value_store.get_kv_server_initializer(
    server_file_path=(
      shared_scratch_root
      .joinpath(config.cluster.kv_store.db_name)
    ),
    server_buffer_size=config.cluster.kv_store.memory_buffer,
  ))
  preloader.register(*text_util.get_scispacy_initalizer(
      scispacy_version=config.parser.scispacy_version,
  ))
  preloader.register(*text_util.get_stopwordlist_initializer(
      stopword_path=config.parser.stopword_list
  ))
  preloader.register(*embedding_util.get_pytorch_device_initalizer(
      disable_gpu=config.sys.disable_gpu,
  ))
  preloader.register(*embedding_util.get_bert_initializer(
      bert_model=config.parser.bert_model,
  ))
  preloader.register(*write_db.get_redis_client_initialzizer(
      host=config.db.address,
      port=config.db.port,
      db=config.db.db_num,
  ))
  # This actual file path will need to be created during the pipeline before use
  preloader.register(*knn_util.get_faiss_index_initializer(
      faiss_index_path=faiss_index_path,
  ))
  if config.pretrained.HasField("sentence_classifier_path"):
    preloader.register(*embedding_util.get_pretrained_model_initializer(
      name="sentence_classifier",
      model_class=SentenceClassifier,
      data_dir=Path(config.pretrained.sentence_classifier_path)
    ))
  dpg.add_global_preloader(client=dask_client, preloader=preloader)

  if config.cluster.clear_checkpoints:
    print("Clearing checkpoint dir")
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()

  def ckpt(name:str, **ckpt_kwargs)->None:
    "Applies checkpointing to the given bag"
    if not config.cluster.disable_checkpoints:
      print("Checkpoint:", name)
      assert name in globals()
      bag = globals()[name]
      assert type(bag) == dbag.Bag
      # Replace bag with result of ckpt, typically with save / load
      globals()[name] = dask_checkpoint.checkpoint(
          bag,
          name=name,
          checkpoint_dir=checkpoint_dir,
          **ckpt_kwargs
      )
    if config.HasField("stop_after_ckpt") and config.stop_after_ckpt == name:
      print("Stopping early.")
      exit(0)


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

  ##############################################################################

  print("Constructing Moliere Database")
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

  sentence_edges_terms = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_interesting_token_keys,
  )
  ckpt("sentence_edges_terms")

  sentence_edges_entities = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_entity_keys,
  )
  ckpt("sentence_edges_entities")

  sentence_edges_mesh = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_mesh_keys,
  )
  ckpt("sentence_edges_mesh")

  sentence_edges_ngrams = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_ngram_keys,
  )
  ckpt("sentence_edges_ngrams")

  sentence_edges_adj = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_adjacent_sentences,
    # We can store only one side of the connection because each sentence will
    # get their own neighbors. Additionally, these should all have the same
    # sort of connections.
    weight_by_tf_idf=False,
    bidirectional=False,
  )
  ckpt("sentence_edges_adj")

  # Join them all together and write to DB
  sentence_edges = dbag.concat([
    sentence_edges_terms,
    sentence_edges_entities,
    sentence_edges_mesh,
    sentence_edges_ngrams,
    sentence_edges_adj,
  ])

  # At this point we have to do the embedding

  sentences_with_embedding = (
      sentences_with_bow
      .map_partitions(
        embedding_util.embed_records,
        # --
        batch_size=config.sys.batch_size,
        text_field="sent_text",
        max_sequence_length=config.parser.max_sequence_length,
      )
  )
  ckpt("sentences_with_embedding")
  final_sentence_records = sentences_with_embedding

  if config.pretrained.HasField("sentence_classifier_path"):
    labeled_sentences = (
        final_sentence_records
        .filter(
          lambda r: r["sent_type"] in SENT_TYPE_SET
        )
    )
    predicted_sentences = (
        final_sentence_records
        .filter(
          lambda r: r["sent_type"] not in SENT_TYPE_SET
        )
        .map_partitions(
          embedding_util.apply_sentence_classifier_to_part,
          # --
          batch_size=config.sys.batch_size
        )
    )
    sentences_with_predicted_types = dbag.concat([
      labeled_sentences,
      predicted_sentences,
    ])
    ckpt("sentences_with_predicted_types")
    final_sentence_records = sentences_with_predicted_types

  hash_and_embedding = (
      final_sentence_records
      .map(
        lambda x: {
          "id": misc_util.hash_str_to_int64(x["id"]),
          "embedding": x["embedding"]
        }
      )
  )
  ckpt("hash_and_embedding")


  # Now we can distribute the knn training
  if not faiss_index_path.is_file():
    print("Training Faiss Index")
    knn_util.train_distributed_knn(
        hash_and_embedding=hash_and_embedding,
        batch_size=config.sys.batch_size,
        num_centroids=config.sentence_knn.num_centroids,
        num_probes=config.sentence_knn.num_probes,
        num_quantizers=config.sentence_knn.num_quantizers,
        bits_per_quantizer=config.sentence_knn.bits_per_quantizer,
        training_sample_prob=config.sentence_knn.training_probability,
        shared_scratch_dir=faiss_index_dir,
        final_index_path=faiss_index_path,
    ).compute()
  else:
    print("Using existing Faiss Index")

  nearest_neighbors_edges = knn_util.nearest_neighbors_network_from_index(
      records=final_sentence_records,
      hash_and_embedding=hash_and_embedding,
      batch_size=config.sys.batch_size,
      num_neighbors=config.sentence_knn.num_neighbors,
  )
  ckpt("nearest_neighbors_edges")

  final_tasks = []
  final_tasks.append(sentence_edges.map_partitions(write_db.write_edges))
  final_tasks.append(
      nearest_neighbors_edges.map_partitions(write_db.write_edges)
  )
  final_tasks.append(sentences_with_bow.map_partitions(write_db.write_records))

  print("Writing everything to redis.")
  dask.compute(final_tasks, sync=True)

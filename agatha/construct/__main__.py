from agatha.config import (
    config_pb2 as cpb,
    proto_util,
)
from agatha.construct import (
    biggraph_util,
    dask_checkpoint,
    dask_process_global as dpg,
    embedding_util,
    file_util,
    ftp_util,
    graph_util,
    knn_util,
    parse_pubmed_xml,
    text_util,
)
from agatha.util import misc_util, database_util
from dask.distributed import Client
from agatha.ml.sentence_classifier import (
    SentenceClassifier,
    LABEL2IDX as SENT_TYPE_SET,
)
from pathlib import Path
import dask
import dask.bag as dbag
import shutil
import json
from datetime import datetime


if __name__ == "__main__":
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running agatha build with the following custom parameters:")
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
  # This directory holds the information necessary to import the mongo database
  _, mongo_data_dir = scratch("mongo_data")

  # export directories
  # This one will hold edge tsv data
  mongo_graph_dir = mongo_data_dir.joinpath("graph")
  mongo_graph_dir.mkdir(parents=True, exist_ok=True)
  # This one will hold sentences stored as json dumps
  mongo_sentences_dir = mongo_data_dir.joinpath("sentences")
  mongo_sentences_dir.mkdir(parents=True, exist_ok=True)

  # Initialize Helper Objects ###
  print("Registering Helper Objects")
  preloader = dpg.WorkerPreloader()
  preloader.register(*database_util.database_initializer(
      address=config.db.address,
      port=config.db.port,
      name=config.db.name,
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

  if config.HasField("cut_date"):
    # This will fail if the cut-date is not a valid string
    datetime.strptime(config.cut_date, "%Y-%m-%d")
    medline_documents = medline_documents.filter(
        lambda r: r["date"] < config.cut_date
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
    weight_by_tf_idf=False,
  )
  ckpt("sentence_edges_terms")

  sentence_edges_entities = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_entity_keys,
    weight_by_tf_idf=False,
  )
  ckpt("sentence_edges_entities")

  sentence_edges_mesh = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_mesh_keys,
    weight_by_tf_idf=False,
  )
  ckpt("sentence_edges_mesh")

  sentence_edges_ngrams = graph_util.record_to_bipartite_edges(
    records=sentences_with_bow,
    get_neighbor_keys_fn=text_util.get_ngram_keys,
    weight_by_tf_idf=False,
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
          "id": misc_util.hash_str_to_int(x["id"]),
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

  hash_and_id = (
      sentences
      .map(lambda rec: {
        "strid": rec["id"],
        "hash": misc_util.hash_str_to_int(rec["id"]),
      })
  )
  write_hash_and_id = database_util.put_bag(
      bag=hash_and_id,
      collection="inverted_index",
      indexed_field_name="hash",
  )
  ckpt("write_hash_and_id")

  nearest_neighbors_edges = knn_util.nearest_neighbors_network_from_index(
      hash_and_embedding=hash_and_embedding,
      inverted_index_collection="inverted_index",
      batch_size=config.sys.batch_size,
      num_neighbors=config.sentence_knn.num_neighbors,
  )
  ckpt("nearest_neighbors_edges")

  all_subgraph_partitions = dbag.concat([
      sentence_edges_terms,
      sentence_edges_entities,
      sentence_edges_mesh,
      sentence_edges_ngrams,
      sentence_edges_adj,
      nearest_neighbors_edges,
  ])
  if config.export_for_mongo:
    print("Writing edges to database dump")
    (
        all_subgraph_partitions
        .map_partitions(graph_util.nxgraphs_to_tsv_edge_list)
        .to_textfiles(f"{mongo_graph_dir}/*.tsv")
    )

    print("Writing sentences to database dump")
    (
        sentences_with_bow
        .map(json.dumps)
        .to_textfiles(f"{mongo_sentences_dir}/*.json")
    )

  if config.HasField("export_with_big_graph_config"):
    print("Processing graph for Pytorch BigGraph")
    pbg_config = biggraph_util.get_biggraph_config(
        Path(config.export_with_big_graph_config)
    )
    biggraph_util.export_graph_for_biggraph(
        pbg_config,
        all_subgraph_partitions
    )




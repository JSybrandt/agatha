from agatha.config import (
    config_pb2 as cpb,
)
from agatha.construct import (
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
from agatha.util import (
    misc_util,
    sqlite3_lookup,
    proto_util,
)
from dask.distributed import Client
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
  _, hash2name_dir = scratch("hash_to_name")
  _, checkpoint_dir = scratch("dask_checkpoints")

  faiss_index_path = faiss_index_dir.joinpath("final.index")

  _, res_data_dir = scratch("processed_data")
  # export directories
  # This one will hold edge tsv data
  res_graph_dir = res_data_dir.joinpath("graph")
  res_graph_dir.mkdir(parents=True, exist_ok=True)
  # This one will hold sentences stored as json dumps
  res_sentence_dir = res_data_dir.joinpath("sentences")
  res_sentence_dir.mkdir(parents=True, exist_ok=True)

  # Initialize Helper Objects ###
  print("Registering Helper Objects")
  preloader = dpg.WorkerPreloader()
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
  if not config.skip_ftp_download:
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
  else:
    print(f"Skipping FTP download, using {download_shared}/*.xml.gz instead")
    assert download_shared.is_dir(), f"Cannot find {download_shared}"
    xml_paths = list(download_shared.glob("*.xml.gz"))
    assert len(xml_paths) > 0, f"No .xml.gz files inside {download_shared}"

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
  ])

  if not config.allow_nonenglish_abstracts:
    medline_documents = medline_documents.filter(
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
      ngram_sample_rate=config.phrases.ngram_sample_rate,
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

  print("Creating Hash2Name Database")
  hash_and_name = (
      sentences
      .map(lambda rec: {
        "name": rec["id"],
        "hash": misc_util.hash_str_to_int(rec["id"]),
      })
  )
  hash2name_db = hash2name_dir.joinpath("hash2name.sqlite3")
  sqlite3_lookup.create_lookup_table(
    record_bag=hash_and_name,
    key_field="hash",
    value_field="name",
    database_path=hash2name_db,
    intermediate_data_dir=hash2name_dir,
    agatha_install_path=config.install_dir,
  )

  # Now we can distribute the knn training
  if not faiss_index_path.is_file():
    print("Training Faiss Index:", faiss_index_path)
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
      hash_and_embedding=hash_and_embedding,
      hash2name_db=hash2name_db,
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
  print("Writing edges to database dump")
  (
      all_subgraph_partitions
      .map_partitions(graph_util.nxgraphs_to_tsv_edge_list)
      .to_textfiles(f"{res_graph_dir}/*.tsv")
  )

  print("Writing sentences to database dump")
  (
      sentences_with_bow
      .map(json.dumps)
      .to_textfiles(f"{res_sentence_dir}/*.json")
  )

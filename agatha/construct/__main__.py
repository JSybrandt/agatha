from agatha.construct import (
    checkpoint,
    dask_process_global as dpg,
    embedding_util,
    file_util,
    ftp_util,
    graph_util,
    knn_util,
    text_util,
    ngram_util,
    construct_config_pb2 as cpb,
    document_pipeline,
)
from agatha.construct.checkpoint import ckpt
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
  _, faiss_index_dir = scratch("faiss_index")
  _, hash2name_dir = scratch("hash_to_name")
  _, checkpoint_dir = scratch("checkpoints")

  # Setup checkpoint
  checkpoint.set_root(checkpoint_dir)
  if config.cluster.disable_checkpoints:
    checkpoint.disable()
  if config.HasField("stop_after_ckpt"):
    checkpoint.set_halt_point(config.stop_after_ckpt)
  if config.cluster.clear_checkpoints:
    checkpoint.clear_all_ckpt()

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

  ##############################################################################
  # BEGIN PIPELINE                                                             #
  ##############################################################################

  # Here's the text data sources

  if config.HasField("medline_xml_dir"):
    document_pipeline.perform_document_independent_tasks(
        config=config,
        documents=document_pipeline.get_medline_documents(config),
        ckpt_prefix="medline"
    )

  if config.HasField("covid_json_dir"):
    document_pipeline.perform_document_independent_tasks(
        config=config,
        documents=document_pipeline.get_covid_documents(config),
        ckpt_prefix="covid"
    )

  # At this point, we are going to recover text sources using the checkpoint
  # module

  ##############################################################################

  parsed_sentences = dbag.concat([
    checkpoint.checkpoint(name, verbose=False)
    for name in checkpoint.get_checkpoints_like("*parsed_sentences")
  ])


  # Perform n-gram mining, introduces a new field "ngrams"
  ngram_sentences = ngram_util.get_frequent_ngrams(
      analyzed_sentences=parsed_sentences,
      max_ngram_length=config.phrases.max_ngram_length,
      min_ngram_support=config.phrases.min_ngram_support,
      min_ngram_support_per_partition=\
          config.phrases.min_ngram_support_per_partition,
      ngram_sample_rate=config.phrases.ngram_sample_rate,
  )
  ckpt("ngram_sentences")

  ngram_edges = graph_util.record_to_bipartite_edges(
    records=ngram_sentences,
    get_neighbor_keys_fn=text_util.get_ngram_keys,
    weight_by_tf_idf=False,
  )
  ckpt("ngram_edges")

  bow_sentences = ngram_sentences.map_partitions(
      text_util.add_bow_to_analyzed_sentence
  )
  ckpt("bow_sentences")

  print("Creating Hash2Name Database")
  hash2name_db = hash2name_dir.joinpath("hash2name.sqlite3")
  sqlite3_lookup.create_lookup_table(
    record_bag=dbag.concat([
      checkpoint.checkpoint(name, verbose=False)
      for name in checkpoint.get_checkpoints_like("*hashed_names")
    ]),
    key_field="hash",
    value_field="name",
    database_path=hash2name_db,
    intermediate_data_dir=hash2name_dir,
    agatha_install_path=config.install_dir,
  )

  # Now we can distribute the knn training
  hashed_embeddings = dbag.concat([
    checkpoint.checkpoint(name, verbose=False)
    for name in checkpoint.get_checkpoints_like("*hashed_embeddings")
  ])

  if not faiss_index_path.is_file():
    print("Training Faiss Index:", faiss_index_path)
    knn_util.train_distributed_knn(
        hash_and_embedding=hashed_embeddings,
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

  knn_edges = knn_util.nearest_neighbors_network_from_index(
      hash_and_embedding=hashed_embeddings,
      hash2name_db=hash2name_db,
      batch_size=config.sys.batch_size,
      num_neighbors=config.sentence_knn.num_neighbors,
  )
  ckpt("knn_edges")

  # Now we can get all edges
  all_edges = dbag.concat([
    checkpoint.checkpoint(name, verbose=False)
    for name in checkpoint.get_checkpoints_like("*_edges")
  ])

  print("Writing edges to database dump")
  (
      all_edges
      .map_partitions(graph_util.nxgraphs_to_tsv_edge_list)
      .to_textfiles(f"{res_graph_dir}/*.tsv")
  )

  print("Writing sentences to database dump")
  (
      bow_sentences
      .map(json.dumps)
      .to_textfiles(f"{res_sentence_dir}/*.json")
  )

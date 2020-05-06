from agatha.construct import (
    checkpoint,
    construct_config_pb2 as cpb,
    dask_process_global as dpg,
    document_pipeline,
    embedding_util,
    file_util,
    ftp_util,
    graph_util,
    knn_util,
    ngram_util,
    semrep_util,
    text_util,
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
from typing import Dict, Any

def setup_directories(config:cpb.ConstructConfig())->Dict[str, Path]:
  # Directory Structure
  """
  {scratch_root_dir}/
    checkpoints/
      ...
    helper_data/
      faiss_index/
      hash_to_name/
      semrep/

  {output_dir}/
    json_dump/
      graph_data/
        ...
      sentence_data/
        ...
  """

  # intermediate dirs
  scratch_root_dir = Path(config.scratch_dir)
  checkpoint_dir = scratch_root_dir.joinpath("checkpoints")
  helper_data_dir = scratch_root_dir.joinpath("helper_data")
  faiss_index_dir = helper_data_dir.joinpath("faiss_index")
  hash2name_dir = helper_data_dir.joinpath("hash_to_name")
  semrep_dir = helper_data_dir.joinpath("semrep")

  # output dirs
  output_dir = Path(config.output_dir)
  output_dump_dir = output_dir.joinpath("json_dump")
  output_graph_dir = output_dump_dir.joinpath("graph_data")
  output_sentence_dir = output_dump_dir.joinpath("sentence_data")

  # For each of the directories specified above
  for val in locals().values():
    if isinstance(val, Path):
      val.mkdir(parents=True, exist_ok=True)

  # Helper Paths
  faiss_index_path = faiss_index_dir.joinpath("final.index")
  hash2name_db = hash2name_dir.joinpath("hash2name.sqlite3")

  # Return all values created in this function
  return locals()

def setup_cluster(config:cpb.ConstructConfig, faiss_index_path:Path)->None:
  # Connect
  if config.cluster.run_locally:
    print("Running on local machine!")
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

  # Initialize Helper Objects on each worker
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
  # If semrep is installed and congiured with agatha
  if (
      config.semrep.HasField("semrep_install_dir")
      and config.semrep.HasField("metamap_install_dir")
  ):
    preloader.register(*semrep_util.get_metamap_server_initializer(
      metamap_install_dir=config.semrep_util.metamap_install_dir
    )
  dpg.add_global_preloader(client=dask_client, preloader=preloader)

def setup_checkpoints(config:cpb.ConstructConfig)->None:
  # Setup checkpoint
  checkpoint.set_root(checkpoint_dir)
  if config.cluster.disable_checkpoints:
    checkpoint.disable()
  if config.HasField("stop_after_ckpt"):
    checkpoint.set_halt_point(config.stop_after_ckpt)
  if config.cluster.clear_checkpoints:
    checkpoint.clear_all_ckpt()


if __name__ == "__main__":
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running agatha build with the following custom parameters:")
  print(config)

  # Adds all setup directories to the current scope
  locals().update(setup_directories(config))
  setup_cluster(config, faiss_index_path)
  setup_checkpoints(config)

  ##############################################################################
  # BEGIN PIPELINE                                                             #
  ##############################################################################


  if config.HasField("medline_xml_dir"):
    document_pipeline.perform_document_independent_tasks(
        config=config,
        documents=document_pipeline.get_medline_documents(config),
        ckpt_prefix="medline",
        semrep_dir=semrep_dir,
    )

  if config.HasField("covid_json_dir"):
    document_pipeline.perform_document_independent_tasks(
        config=config,
        documents=document_pipeline.get_covid_documents(config),
        ckpt_prefix="covid",
        semrep_dir=semrep_dir,
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
  hash_name_kv = (
    dbag.concat([
      checkpoint.checkpoint(name, verbose=False)
      for name in checkpoint.get_checkpoints_like("*hashed_names")
    ])
    .map(
      lambda r: {"key": r["hash"], "value": r["name"]}
    )
  )
  sqlite3_lookup.create_lookup_table(
    key_value_records=hash_name_kv,
    result_database_path=hash2name_db,
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
  print("Writing graph json dump")
  graph_kv = (
    dbag.concat([
      checkpoint.checkpoint(name, verbose=False)
      for name in checkpoint.get_checkpoints_like("*_edges")
    ])
    .map_partitions(graph_util.nxgraphs_to_kv)
  )
  sqlite3_lookup.export_key_value_records(
      key_value_records=graph_kv,
      export_dir=output_graph_dir,
  )

  print("Writing sentence json dump")
  sentence_kv = bow_sentences.map(
      lambda r: dict(
        key=r["id"],
        value=dict(
          bow=r["bow"],
          sent_text=r["sent_text"],
        )
      )
  )
  sqlite3_lookup.export_key_value_records(
      key_value_records=sentence_kv,
      export_dir=output_sentence_dir
  )

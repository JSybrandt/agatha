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
from dask.distributed import Client, LocalCluster
import dask.bag as dbag
import dask
from pathlib import Path
import faiss
from copy import copy
from typing import Dict, Any, Callable


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

    print("Splitting sentences.")
    pubmed_sentences = dbag.from_delayed([
      dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
        xml_path=p,
        local_scratch=download_local
      )
      for p in xml_paths
    ]).filter(lambda r: r["language"]=="eng"
    ).map(text_util.split_sentences,
    ).flatten()

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
  else:
    pubmed_sent_w_ent = file_util.load(out_sent_w_ent)

  pubmed_sent_w_ent = pubmed_sent_w_ent.random_sample(0.01)

  print("Training KNN for sentence embeddings.")
  _, tmp_faiss_index_dir = mk_scratch("tmp_faiss_index")
  trained_knn_path = knn_util.train_distributed_knn_from_text_fields(
      text_records=pubmed_sent_w_ent,
      id_fn=lambda r:f"{r['pmid']}:{r['version']}:{r['sent_idx']}",
      text_field="sent_text",
      scibert_data_dir=config.parser.scibert_data_dir,
      batch_size=config.parser.batch_size,
      num_centroids=config.sentence_knn.num_centroids,
      num_probes=config.sentence_knn.num_probes,
      num_quantizers=config.sentence_knn.num_quantizers,
      bits_per_quantizer=config.sentence_knn.bits_per_quantizer,
      training_sample_prob=config.sentence_knn.training_probability,
      shared_scratch_dir=tmp_faiss_index_dir,
  )

from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
    dask_config,
)
from pymoliere.util import file_util, ftp_util
from pymoliere.construct import parse_pubmed_xml, text_util
from dask.distributed import Client, LocalCluster
import dask.bag as dbag
import dask
from pathlib import Path

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

  _, out_sent = file_util.prep_scratches(
    local_scratch_root=local_scratch_root,
    shared_scratch_root=shared_scratch_root,
    task_name="pubmed_sentences",
  )
  if file_util.is_result_saved(out_sent):
    pubmed_sentences = file_util.load(out_sent)
  else:
    print("Splitting sentences.")
    print("\t- Initializing helper object")
    dask_client.run(
        text_util.init_split_sentences,
        # --
        scispacy_version=config.parser.scispacy_version,
    )
    print("\t- Done!")
    pubmed_sentences = dbag.from_delayed([
      dask.delayed(parse_pubmed_xml.parse_pubmed_xml)(
        xml_path=p,
        local_scratch=download_local
      )
      for p in xml_paths
    ]).filter(
        lambda r: r["language"]=="eng"
    ).map(
        text_util.split_sentences,
        # --
        text_fields=["title", "abstract"],
    ).flatten(
    ).persist()
    print("\t- Saving...")
    file_util.save(pubmed_sentences, out_sent)

  _, out_sent_w_ent = file_util.prep_scratches(
    local_scratch_root=local_scratch_root,
    shared_scratch_root=shared_scratch_root,
    task_name="pubmed_sent_w_ent",
  )
  if file_util.is_result_saved(out_sent_w_ent):
    pubmed_sent_w_ent = file_util.load(out_sent_w_ent)
  else:
    print("Analyzing each document.")
    print("\t- Initializing helper object")
    dask_client.run(
        text_util.init_analyze_sentence,
        # --
        scispacy_version=config.parser.scispacy_version,
        scibert_dir=config.parser.scibert_data_dir,
    )
    print("\t- Done!")
    pubmed_sent_w_ent = pubmed_sentences.map(
        text_util.analyze_sentence,
        # --
        text_field="sentence",
    )
    print("\t- Saving...")
    file_util.save(pubmed_sent_w_ent, out_sent_w_ent)

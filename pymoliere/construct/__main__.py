from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
    dask_config,
)
from pymoliere.util import file_util, ftp_util
from pymoliere.construct.parse_pubmed_xml import parse_pubmed_xml
from pymoliere.construct.text_util import (
    setup_scispacy,
    split_sentences,
    add_entitites,
)
from dask.distributed import Client
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
  cluster_address = f"{config.cluster.address}:{config.cluster.port}"

  # Configure Dask
  print("Configuring Dask, attaching to cluster")
  print(f"\t{cluster_address}")
  dask_config.set_local_tmp(local_scratch_root)
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

  pubmed = dbag.from_delayed([
    dask.delayed(parse_pubmed_xml)(
      xml_path=p,
      local_scratch=download_local
    )
    for p in xml_paths
  ])

  pubmed_eng = pubmed.filter(lambda r: r["language"]=="eng")

  pubmed_sentences = pubmed_eng.map(
      split_sentences,
      # --
      text_fields=["title", "abstract"],
      scispacy_version=config.parser.scispacy_version,
  ).flatten().repartition(500)

  # pubmed_sentences = pubmed_sentences.persist()
  # count = pubmed_sentences.count().compute()
  # print(f"Found {count} pubmed sentences")
  # example = pubmed_sentences.take(5)
  # print("Here's an example:")
  # print(example)

  _, out_shared = file_util.prep_scratches(
    local_scratch_root=local_scratch_root,
    shared_scratch_root=shared_scratch_root,
    task_name="pubmed_sentences",
  )
  file_util.save(pubmed_sentences, out_shared)


  # pubmed_sent_with_ent = pubmed_sentences.map(
      # add_entitites,
      # # --
      # text_field="sentence",
      # nlp=scispacy,
  # )
  # pubmed_sent_with_ent.persist()

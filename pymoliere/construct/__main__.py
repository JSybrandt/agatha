from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
    dask_config,
)
from pymoliere.util import file_util, ftp_util, pipeline_operator
from pymoliere.construct.parse_pubmed_xml import ParsePubmedOperator
from dask.distributed import Client
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
  dclient = Client(address=cluster_address)
  dclient.restart()

  # Configure pipeline defaults
  pipeline_operator.DEFAULTS["shared_scratch_root"] = shared_scratch_root
  pipeline_operator.DEFAULTS["local_scratch_root"] = local_scratch_root

  # READY TO GO!

  print("Retrieving PubMed")
  pubmed_xml_dir = file_util.prep_scratch_subdir(
      scratch_root=shared_scratch_root,
      dir_name="download_pubmed",
  )
  with ftp_util.ftp_connect(
      address=config.ftp.address,
      workdir=config.ftp.workdir,
  ) as conn:
    xml_paths = ftp_util.ftp_retreive_all(
        conn=conn,
        pattern="^.*\.xml\.gz$",
        directory=pubmed_xml_dir,
        show_progress=True,
    )

  pubmed = ParsePubmedOperator(
      shared_xml_gz_paths=xml_paths,
      name="parse_pubmed",
  ).eval()

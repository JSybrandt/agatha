from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
    dask_config,
)
from pymoliere.construct.ftp_util import (
    ftp_connect,
    ftp_retreive_all,
)
from pymoliere.construct.file_util import (
    prep_scratch_subdir,
    load_or_construct_dask_df
)
from pymoliere.construct.parse_pubmed_xml import parse_pubmed_xml
from dask.delayed import delayed
from dask.distributed import (
    Client,
)
import dask.dataframe as ddf
from pathlib import Path
from typing import List


def download_pubmed_xml_gz(
  ftp_address:str,
  ftp_workdir:str,
  download_dir:Path,
  show_progress:bool,
)->List[Path]:
  "Downloads ftp files for pubmed release (or uses cached version)"
  with ftp_connect(address=ftp_address, workdir=ftp_workdir) as conn:
    return ftp_retreive_all(
        conn=conn,
        pattern="^.*\.xml\.gz$",
        local_dir=download_dir,
        show_progress=show_progress,
    )


def main() -> None:
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere build with the following custom parameters:")
  print(config)

  # Checks
  print("Performing config checks")
  shared_scratch_dir = Path(config.cluster.shared_scratch)
  shared_scratch_dir.mkdir(parents=True, exist_ok=True)
  assert shared_scratch_dir.is_dir()
  local_scratch_dir = Path(config.cluster.local_scratch)
  local_scratch_dir.mkdir(parents=True, exist_ok=True)
  assert local_scratch_dir.is_dir()
  cluster_address = f"{config.cluster.address}:{config.cluster.port}"
  intermediate_artifacts = prep_scratch_subdir(shared_scratch_dir, "artifacts")

  # Configure Dask
  print("Configuring Dask, attaching to cluster")
  print(f"\t{cluster_address}")
  dask_config.set_local_tmp(local_scratch_dir)
  dclient = Client(address=cluster_address)

  # READY TO GO!

  print("Retrieving PubMed")
  shared_xml_paths = download_pubmed_xml_gz(
      ftp_address=config.ftp.address,
      ftp_workdir=config.ftp.workdir,
      download_dir=prep_scratch_subdir(shared_scratch_dir, "pubmed"),
      show_progress=True,
  )

  # Parse each file
  def do_parse():
    local_dfs = [
        delayed(parse_pubmed_xml)(
          shared_xml_gz_path=p,
          local_scratch_dir=local_scratch_dir,
        )
        for p in shared_xml_paths
    ]
    return ddf.from_delayed(local_dfs)
  pubmed_df = load_or_construct_dask_df(
    construct_fn=do_parse,
    paraquet_dir=prep_scratch_subdir(intermediate_artifacts, "pubmed_parsed")
  )
  print(pubmed_df.dtypes)


if __name__ == "__main__":
  main()

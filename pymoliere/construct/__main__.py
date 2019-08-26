from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct.ftp_util import (
    ftp_connect,
    ftp_retreive_all,
)
from pymoliere.construct.parse_pubmed_xml import parse_pubmed_xml
from dask import (
    compute
)
from dask.delayed import delayed
from dask.distributed import (
    Client,
    wait
)
import dask.bag as dbag
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


def copy_to_local_scratch(src:Path, local_scratch:Path)->Path:
  assert src.is_file()
  assert local_scratch.is_dir()
  dest  = local_scratch.joinpath(src.name)
  dest.write_bytes(src.read_bytes())
  return dest


def main() -> None:
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere build with the following custom parameters:")
  print(config)

  # Checks
  shared_scratch_dir = Path(config.cluster.shared_scratch)
  assert shared_scratch_dir.is_dir()

  print(f"Attaching to cluster")
  cluster_address = f"{config.cluster.address}:{config.cluster.port}"
  print(f"\t{cluster_address}")
  dclient = Client(address=cluster_address)

  print("Retrieving PubMed")
  pubmed_shared_scratch_dir = shared_scratch_dir.joinpath("pubmed")
  pubmed_shared_scratch_dir.mkdir(parents=True, exist_ok=True)
  xml_paths = download_pubmed_xml_gz(
      ftp_address=config.ftp.address,
      ftp_workdir=config.ftp.workdir,
      download_dir=pubmed_shared_scratch_dir,
      show_progress=True,
  )

  # Transition to distributed system
  print("Parsing xml files")
  xml_paths = dbag.from_sequence(xml_paths, partition_size=1)
  local_paths = xml_paths.map(
      copy_to_local_scratch,
      local_scratch=Path(config.cluster.local_scratch),
  )
  articles_per_file = local_paths.map(parse_pubmed_xml)
  pubmed_articles = articles_per_file.flatten()
  print("Found", pubmed_articles.count().compute())

if __name__ == "__main__":
  main()

# PREPARE TRAINING DATA FOR SENTENCE CLASSIFIER
# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import (
    dask_process_global as dpg,
    embedding_util,
    file_util,
    ftp_util,
    parse_pubmed_xml,
    text_util,
)
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask
import dask.bag as dbag
from pymoliere.util.misc_util import Record
from pymoliere.ml.util.sentence_classifier import (
    LABEL2IDX,
)
from pymoliere.construct import dask_checkpoint
from typing import Iterable


def filter_and_pick(records:Iterable[Record])->Iterable[Record]:
  return list(map(
      lambda r: {
        "sent_text": r["sent_text"],
        "sent_type": r["sent_type"],
        "sent_ratio": r["sent_idx"] / r["sent_total"]
      },
      filter(
        lambda r: r["sent_type"] in LABEL2IDX,
        records
      )
  ))


if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere sentence_classifier with the following parameters:")
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
    cluster = LocalCluster(n_workers=1)
    dask_client = Client(cluster)
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address, heartbeat_interval=500)
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
  _, download_shared = scratch("download_pubmed")
  _, checkpoint_dir = scratch("dask_checkpoints")
  # By sharing the same checkpoint dir as moliere, the prep can take advantage
  # of partial results

  if config.cluster.clear_checkpoints:
    print("Clearing checkpoint dir")
    shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir()

  def ckpt(name:str)->None:
    "Applies checkpointing to the given bag"
    if not config.cluster.disable_checkpoints:
      assert name in globals()
      assert type(globals()[name]) == dbag.Bag
      globals()[name] = dask_checkpoint.checkpoint(
          globals()[name],
          name=name,
          checkpoint_dir=checkpoint_dir,
      )

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

  preloader = dpg.WorkerPreloader()
  preloader.register(*embedding_util.get_scibert_initializer(
      scibert_data_dir=config.parser.scibert_data_dir,
      disable_gpu=config.ml.disable_gpu,
  ))
  dpg.add_global_preloader(client=dask_client, preloader=preloader)

  ##############################################################################
  # READY TO GO!

  ## DUPLICATED FROM MAIN MOLIERE
  medline_documents = dbag.from_delayed([
    dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
      xml_path=p,
    )
    for p in xml_paths
  ]).filter(
    # Only take the english ones
    lambda r: r["language"]=="eng"
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

  ## END OF DUPLICATED SECTION
  ml_labeled_sentences = sentences.map_partitions(filter_and_pick)
  ckpt("ml_labeled_sentences")

  ml_embedded_labeled_sentences = ml_labeled_sentences.map_partitions(
      embedding_util.embed_records,
      # --
      batch_size=config.ml.batch_size,
      text_field="sent_text",
      max_sequence_length=config.parser.max_sequence_length,
  )
  ckpt("ml_embedded_labeled_sentences")

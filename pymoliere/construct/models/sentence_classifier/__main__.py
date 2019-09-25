# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import (
    dask_checkpoint,
    dask_process_global as dpg,
    file_util,
    ftp_util,
    parse_pubmed_xml,
    text_util,
    embedding_util,
)
from pymoliere.util.misc_util import Record
from pathlib import Path
from dask.distributed import (
  Client,
  LocalCluster,
)
import dask
import dask.bag as dbag


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

  def mk_scratch(task_name):
    "returns local, shared"
    return file_util.prep_scratches(
      local_scratch_root=local_scratch_root,
      shared_scratch_root=shared_scratch_root,
      task_name=task_name,
    )

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

  print("Registering Helper Objects")
  def prepare_dask_process_global():
    dpg.clear()
    dpg.register(*text_util.get_stopwordlist_initializer(
        stopword_path=config.parser.stopword_list
    ))
    dpg.register(*embedding_util.get_scibert_initializer(
        scibert_data_dir=config.parser.scibert_data_dir,
        disable_gpu=config.sys.disable_gpu,
    ))
  dask_client.run(prepare_dask_process_global)

  # Prepping all scratch dirs ###
  print("Prepping scratch directories")
  _, download_shared = mk_scratch("download_pubmed")
  _, checkpoint_dir = mk_scratch("sent_classifier_checkpoints")

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

  ##############################################################################
  # READY TO GO!

  def pluck_relevant_sentence_fields(sent_rec:Record)->Record:
    fields = [
        "sent_text",
        "sent_type",
        "sent_idx",
        "id"
    ]
    return {k:sent_rec[k] for k in fields}

  # Parse xml-files per-partition
  labeled_sentences = dbag.from_delayed([
      dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
          xml_path=p,
      )
      for p in xml_paths
  ]).filter(
      lambda r: r["language"]=="eng"
  ).map(
      text_util.split_sentences,
      # --
      min_sentence_len=config.parser.min_sentence_len,
      max_sentence_len=config.parser.max_sentence_len,
  ).flatten(
  ).filter(
      lambda r: r["sent_type"] in {
        "abstract:background",
        "abstract:conclusions",
        "abstract:methods",
        "abstract:objective",
        "abstract:results",
      }
  ).map(pluck_relevant_sentence_fields)

  labeled_sentences = dask_checkpoint.checkpoint(
      labeled_sentences,
      name="labeled_sentences",
      checkpoint_dir=checkpoint_dir,
  )

  embedded_sentences = labeled_sentences.map_partitions(
      embedding_util.embed_records,
      # --
      batch_size=config.sys.batch_size,
      text_field="sent_text",
      max_sequence_length=config.parser.max_sequence_length,
  )
  embedded_sentences = dask_checkpoint.checkpoint(
      embedded_sentences,
      name="embedded_sentences",
      checkpoint_dir=checkpoint_dir,
  )

  print("Running!!!")
  tasks = [
      embedded_sentences,
  ] + dask_checkpoint.get_checkpoint_tasks()
  tasks = dask.optimize(tasks)
  dask_client.compute(tasks, sync=True)

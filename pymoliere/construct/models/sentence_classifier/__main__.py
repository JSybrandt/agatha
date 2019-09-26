# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import (
    file_util,
    ftp_util,
    parse_pubmed_xml,
    text_util,
)
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask
import dask.bag as dbag
from pytorch_transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.nn import functional as F
from pymoliere.construct.models import train_model
import gzip
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

IDX2LABEL = [
  'abstract:background',
  'abstract:conclusions',
  'abstract:methods',
  'abstract:objective',
  'abstract:results',
]
LABEL2IDX = {l: i for i, l in enumerate(IDX2LABEL)}
NUM_LABELS = len(IDX2LABEL)
SCIBERT_OUTPUT_DIM = 768

class SentenceClassifier(torch.nn.Module):
  def __init__(self, scibert_data_dir:Path):
    super(SentenceClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(scibert_data_dir)
    self.l1 = torch.nn.Linear(SCIBERT_OUTPUT_DIM, 256)
    self.l2 = torch.nn.Linear(256, NUM_LABELS)
    self.linear = [
        self.l1, self.l2,
    ]
    # freeze bert layers
    for param in self.bert.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.bert(x)[-1]
    # for all but the last
    for l in self.linear[:-1]:
      x = F.relu(l(x))
    # for the last
    return self.linear[-1](x)


def get_batch_to_tensors(scibert_data_dir:Path)->train_model.ToTensorFn:
  tok = BertTokenizer.from_pretrained(scibert_data_dir)
  def batch_to_tensors(sents, labels):
      x = [torch.tensor(tok.encode(s)) for s in sents]
      x = pad_sequence(x, batch_first=True)
      y = torch.LongTensor(labels)
      return x, y
  return batch_to_tensors


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
  _, labeled_sentence_shared = scratch("labeled_sentences")

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

  # Parse xml-files per-partition
  if False and not file_util.is_result_saved(labeled_sentence_shared):
    labeled_sentences = dbag.from_delayed([
        dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
            xml_path=p,
        )
        for p in xml_paths
    ]).filter(
        lambda r: r["language"]=="eng"
    ).map_partitions(
        text_util.split_sentences,
        # --
        min_sentence_len=config.parser.min_sentence_len,
        max_sentence_len=config.parser.max_sentence_len,
    ).filter(
        lambda r: r["sent_type"] in LABEL2IDX
    )
    print("Saving parsed sentences")
    file_util.save(labeled_sentences, labeled_sentence_shared)
  else:
    print("Using pre-split sentences")

  print("Loading")
  sentences = []
  labels = []
  for file_path in tqdm(list(labeled_sentence_shared.iterdir())[:10]):
    if file_path.suffix == ".gz":
      with gzip.open(str(file_path)) as f:
        for line in f:
          record = json.loads(line)
          if record["sent_type"] in LABEL2IDX:
            sentences.append(record["sent_text"])
            labels.append(LABEL2IDX[record["sent_type"]])

  assert len(sentences) == len(labels)
  print(f"Loaded {len(sentences)} sentences.")

  print("Prepping model")
  if torch.cuda.is_available() and not config.ml.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier(scibert_data_dir=config.parser.scibert_data_dir)
  batch_to_tensor_fn = get_batch_to_tensors(
      scibert_data_dir=config.parser.scibert_data_dir
  )
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
  )

  print("Beginning Training")
  train_model.train_classifier(
      model=model,
      device=device,
      loss_fn=loss_fn,
      optimizer=optimizer,
      num_epochs=config.ml.num_epochs,
      batch_size=config.ml.batch_size,
      data=sentences,
      labels=labels,
      validation_ratio=config.ml.validation_ratio,
      batch_to_tensor_fn=batch_to_tensor_fn,
  )

  torch.save(model.state_dict(), config.output)


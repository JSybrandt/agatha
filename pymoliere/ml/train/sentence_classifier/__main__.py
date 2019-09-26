# TRAIN SENTENCE CLASSIFIER
# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pymoliere.construct import file_util
from pathlib import Path
from dask.distributed import Client, LocalCluster
import dask
import dask.bag as dbag
from pytorch_transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.nn import functional as F
from pymoliere.ml.train import train_model
import gzip
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from pymoliere.ml.util.sentence_classifier import (
    IDX2LABEL,
    LABEL2IDX,
    NUM_LABELS,
    SCIBERT_OUTPUT_DIM,
)


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

  labeled_sentence_dir = Path(
      config.cluster.shared_scratch
  ).joinpath(
      "dask_checkpoints"
  ).joinpath(
      "labeled_sentences"
  )

  if not file_util.is_result_saved(labeled_sentence_dir):
    raise Exception(f"Could not find prepared data in {labeled_sentence_dir}")

  print("Loading")
  sentences = []
  labels = []
  for file_path in tqdm(list(labeled_sentence_dir.iterdir())[:10]):
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
    device = torch.device("cuda:0")
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

  # if torch.cuda.device_count() > 1:
    # print("Enabling  mutli-gpu training")
    # model = nn.DataParallel(model)
    # config.ml.batch_size *= torch.cuda.device_count()

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


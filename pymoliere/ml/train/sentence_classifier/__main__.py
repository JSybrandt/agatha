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
from pymoliere.ml.train import train_model, evaluate_model
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
import numpy as np


class SentenceClassifier(torch.nn.Module):
  def __init__(self):
    super(SentenceClassifier, self).__init__()
    self.l1 = torch.nn.Linear(SCIBERT_OUTPUT_DIM+1, 512)
    self.l2 = torch.nn.Linear(512, 256)
    self.l3 = torch.nn.Linear(256, NUM_LABELS)
    self.linear = [
        self.l1, self.l2, self.l3,
    ]

  def forward(self, x):
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

  print("Loading")
  embedding_data_dir = Path(
      config.cluster.shared_scratch
  ).joinpath(
      "dask_checkpoints"
  ).joinpath(
      "ml_embedded_labeled_sentences"
  )
  records = file_util.load_to_memory(embedding_data_dir)

  print("Converting")
  data = [
      np.append(rec["embedding"], rec["sent_ratio"])
      for rec in records
  ]

  labels = [
      LABEL2IDX[rec["sent_type"]]
      for rec in records
  ]

  print("Prepping model")
  if torch.cuda.is_available() and not config.ml.disable_gpu:
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier()
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
  )

  # if torch.cuda.device_count() > 1:
    # print("Enabling  mutli-gpu training")
    # model = nn.DataParallel(model)
    # config.ml.batch_size *= torch.cuda.device_count()

  if not Path(config.model_output_path).is_file():
    print("Beginning Training")
    train_model.train_classifier(
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=config.ml.num_epochs,
        batch_size=config.ml.batch_size,
        data=data,
        labels=labels,
        validation_ratio=config.ml.validation_ratio,
        batch_to_tensor_fn=lambda x, y: (torch.FloatTensor(x), torch.LongTensor(y)),
    )
    print("Saving model")
    torch.save(model.state_dict(), config.model_output_path)
  else:
    print("Loading Model")
    model.load_state_dict(torch.load(config.model_output_path))

  if config.HasField("evaluation_output_dir"):
    print("Evaluation")
    evaluate_model.evaluate_multiclass_model(
        model=model,
        device=device,
        batch_size=config.ml.batch_size,
        data_batch_to_tensor_fn=lambda x: torch.FloatTensor(x),
        data=data,
        labels=labels,
        metadata=[r["sent_text"] for r in records],
        class_names=IDX2LABEL,
        output_dir=Path(config.evaluation_output_dir),
    )


# TRAIN SENTENCE CLASSIFIER
# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from collections import namedtuple
from pathlib import Path
from pymoliere.construct import file_util
from pymoliere.ml import train_model, evaluate_model
from pymoliere.util.misc_util import Record
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from typing import List, Tuple, Any
import numpy as np
import pickle
import torch
import dask
from dask.distributed import Client, LocalCluster


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

TrainingData = namedtuple(
    "TrainingData",
    ["dense_data", "label", "date"]
)


class SentenceClassifier(torch.nn.Module):
  def __init__(self):
    super(SentenceClassifier, self).__init__()
    self.l1 = torch.nn.Linear(SCIBERT_OUTPUT_DIM+1, 512)
    self.r1 = torch.nn.ReLU(inplace=True)
    self.l2 = torch.nn.Linear(512, 256)
    self.r2 = torch.nn.ReLU(inplace=True)
    self.l3 = torch.nn.Linear(256, NUM_LABELS)
    self.soft = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.l1(x)
    x = self.r1(x)
    x = self.l2(x)
    x = self.r2(x)
    x = self.l3(x)
    x = self.soft(x)
    return x

def record_to_sentence_classifier_input(record:Record)->torch.FloatTensor:
  return torch.FloatTensor(
    np.append(
      record["embedding"],
      record["sent_idx"] / float(record["sent_total"])
    )
  )


def sentence_classifier_output_to_labels(
    batch_logits:torch.FloatTensor,
)->List[str]:
  _, batch_predictions = torch.max(batch_logits, 1)
  batch_predictions = batch_predictions.detach().cpu().numpy().tolist()
  return [IDX2LABEL[i] for i in batch_predictions]


def load_training_data_from_ckpt(ckpt_dir:Path)->List[Record]:
  def part_to_data(part):
    res = []
    for record in part:
      if record["sent_type"] in LABEL2IDX:
        res.append(TrainingData(
          dense_data=record_to_sentence_classifier_input(record),
          label=LABEL2IDX[record["sent_type"]],
          date=record["date"],
        ))
    return res
  print("Loading from pymoliere checkpoint.")
  return (
      file_util.load(ckpt_dir)
      .map_partitions(part_to_data)
      .compute()
  )


def create_or_load_training_data(
    shared_scratch:Path,
    data_path:Path,
)->Tuple[List[Any], List[Any], List[Any]]:
  "Returns train, validation, test"
  training_data_scratch = (
      shared_scratch
      .joinpath("models")
      .joinpath("sentence_classifier")
  )
  training_data_scratch.mkdir(parents=True, exist_ok=True)
  data_path = training_data_scratch.joinpath("data.pkl")

  if not data_path.is_file():
    sent_w_emb_ckpt_dir = (
        shared_scratch
        .joinpath("dask_checkpoints")
        .joinpath("sentences_with_embedding")  # name from construct.__main__
    )
    if not file_util.is_result_saved(sent_w_emb_ckpt_dir):
      print("Error, failed to find `sentences_with_embedding` checkpoint")
      print("Run pymoliere.construct with:")
      print("`--stop_after_ckpt sentences_with_embedding`")
      exit(1)
    # Load what we need from ckpt
    data = load_training_data_from_ckpt(sent_w_emb_ckpt_dir)
    data.sort(key=lambda x: x.date)
    test_set_size = int(len(data)*config.test_set_ratio)
    validation_set_size = int(
        (len(data) - test_set_size) * config.validation_set_ratio
    )
    idx_2 = len(data) - test_set_size
    idx_1 = idx_2 - validation_set_size
    idx_1 = len(data) - test_set_size - validation_set_size
    training_data = data[:idx_1]
    validation_data = data[idx_1:idx_2]
    test_data = data[idx_2:]
    with open(data_path, 'wb') as f:
      pickle.dump(
          {
            "train": training_data,
            "validation": validation_data,
            "test": test_data,
          },
          f,
          protocol=4,
      )
  else:
    with open(data_path, 'rb') as f:
      data = pickle.load(f)
    training_data = data["train"]
    validation_data = data["validation"]
    test_data = data["test"]
  print("Last Training Date:", training_data[-1].date)
  print("Last Validation Date:", validation_data[-1].date)
  print("Last Test Date:", test_data[-1].date)
  print(f"{len(training_data)} - {len(validation_data)} - {len(test_data)}")
  return training_data, validation_data, test_data


if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere sentence_classifier with the following parameters:")
  print(config)

  # if config.cluster.run_locally:
    # print("Running on local machine!")
    # cluster = LocalCluster()
    # dask_client = Client(cluster)
  # else:
    # cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    # print("Configuring Dask, attaching to cluster")
    # print(f"\t- {cluster_address}")
    # dask_client = Client(address=cluster_address)
  # if config.cluster.restart:
    # print("\t- Restarting cluster...")
    # dask_client.restart()
  # print(f"\t- Running on {len(dask_client.nthreads())} machines.")

  shared_scratch = Path(config.shared_scratch)
  training_data_scratch = (
      shared_scratch
      .joinpath("models")
      .joinpath("sentence_classifier")
  )
  training_data_scratch.mkdir(parents=True, exist_ok=True)
  data_path = training_data_scratch.joinpath("data.pkl")

  training_data, validation_data, test_data = create_or_load_training_data(
      shared_scratch=shared_scratch,
      data_path=data_path
  )

  print("Prepping model")
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier()
  # if torch.cuda.device_count() and not config.sys.disable_gpu:
    # print("Going to two gpu")
    # model = torch.nn.DataParallel(model)
  model.to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
  )

  if config.force_retrain or not Path(config.model_output_path).is_file():
    print("Beginning Training")
    train_model.train_classifier(
        training_data=[x.dense_data for x in training_data],
        training_labels=[x.label for x in training_data],
        validation_data=[x.dense_data for x in validation_data],
        validation_labels=[x.label for x in validation_data],
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=config.sys.num_epochs,
        batch_size=config.sys.batch_size,
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
        batch_size=config.sys.batch_size,
        data=[x.dense_data for x in test_data],
        labels=[x.label for x in test_data],
        class_names=IDX2LABEL,
    )


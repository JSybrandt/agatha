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
from typing import List, Tuple, Any, Iterable
import numpy as np
import torch
import dask
from dask.distributed import Client
from pymoliere.construct.dask_checkpoint import checkpoint


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

################################################################################
# Model ########################################################################
################################################################################

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

################################################################################
# Exported Functions ###########################################################
################################################################################

def record_to_sentence_classifier_input(record:Record)->torch.Tensor:
  return record_to_training_tuple(record).dense_data

def sentence_classifier_output_to_labels(
    batch_logits:torch.FloatTensor,
)->List[str]:
  _, batch_predictions = torch.max(batch_logits, 1)
  batch_predictions = batch_predictions.detach().cpu().numpy().tolist()
  return [IDX2LABEL[i] for i in batch_predictions]

################################################################################
# Helper Functions #############################################################
################################################################################

def record_to_training_tuple(record:Record)->TrainingData:
  """
  Converts dict to named tuple. We don't actually need to load most of the
  record.
  """
  return TrainingData(
      dense_data=torch.FloatTensor(
        np.append(
          record["embedding"],
          record["sent_idx"] / float(record["sent_total"]),
        )
      ),
      label=LABEL2IDX[record["sent_type"]],
      date=record["date"],
  )


def record_is_labeled(record:Record)->bool:
  return record["sent_type"] in LABEL2IDX


def filter_sentences_with_embedding(
    records:Iterable[Record]
)->Iterable[TrainingData]:
  """
  Converts a partition of records to training data tuples, provided they have a
  useful training label.
  """
  res = []
  for record in records:
    if record_is_labeled(record):
      res.append(record_to_training_tuple(record))
  return res


def get_boundary_dates(
    dates:List[str],
    test_ratio:float,
    validation_ratio:float,
)->Tuple[str, str]:
  """
  Given a set of dates, returns two to partition the data into
  training/validation/test
  """
  assert test_ratio < 1
  assert validation_ratio < 1
  assert test_ratio + validation_ratio < 1
  dates.sort()
  all_data = len(dates)
  test_size = int(all_data * test_ratio)
  validation_size = int((all_data - test_size) * validation_ratio)
  train_size = all_data - test_size - validation_size
  return dates[train_size], dates[train_size+validation_size]

################################################################################

if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere sentence_classifier with the following parameters:")
  print(config)

  # Potential cluster
  if config.cluster.run_locally or config.cluster.address == "localhost":
    print("Running on local machine!")
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
  if config.cluster.restart:
    print("\t- Restarting cluster...")
    dask_client.restart()

  # Prep scratches
  shared_scratch = Path(config.shared_scratch)
  load_ckpt_dir = (
      shared_scratch
      .joinpath("dask_checkpoints")
  )
  # We're going to store model-specific checkpoints seperatly
  checkpoint_dir = (
      shared_scratch
      .joinpath("models")
      .joinpath("sentence_classifier")
      .joinpath("dask_checkpoints")
  )
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  model_path = (
      shared_scratch
      .joinpath("models")
      .joinpath("sentence_classifier")
      .joinpath("model.pt")
  )

  # All data, this is the checkpoint we depend on
  sentences_with_embedding = file_util.load(
      load_ckpt_dir.joinpath("sentences_with_embedding")
  )
  # Get only results with labels, store at TrainingData tuples
  sentence_classifier_all_data = sentences_with_embedding.map_partitions(
      filter_sentences_with_embedding
  )
  checkpoint(
      sentence_classifier_all_data,
      name="sentence_classifier_all_data",
      checkpoint_dir=checkpoint_dir,
  )

  if not file_util.is_result_saved(
      checkpoint_dir.joinpath("training_data")
  ):
    print("Finding the training data!")
    print("Getting Dates")
    sample_of_dates = (
        sentence_classifier_all_data
        .random_sample(0.05)
        .map(lambda x: x.date)
        .compute()
    )
    val_date, test_date = get_boundary_dates(
        dates=sample_of_dates,
        validation_ratio=config.validation_set_ratio,
        test_ratio=config.test_set_ratio,
    )
    print("Training goes up to ", val_date)
    print(f"Validation is from {val_date} to {test_date}")
    print("Testing is after", test_date)
    save_training_data = file_util.save(
        bag=(
          sentence_classifier_all_data
          .filter(lambda x: x.date < val_date)
        ),
        path=checkpoint_dir.joinpath("training_data")
    )
    save_validation_data = file_util.save(
        bag=(
          sentence_classifier_all_data
          .filter(lambda x: val_date <= x.date < test_date)
        ),
        path=checkpoint_dir.joinpath("validation_data")
    )
    save_test_data = file_util.save(
        bag=(
          sentence_classifier_all_data
          .filter(lambda x: test_date <= x.date)
        ),
        path=checkpoint_dir.joinpath("test_data")
    )
    print("Filtering and saving training/validation/testing data.")
    dask.compute(save_training_data, save_validation_data, save_test_data)

  # Training data is ready, time to go!
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

  loss_fn = nn.NLLLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
  )


  if config.force_retrain or not model_path.is_file():
    # Get your data!
    training_data = file_util.load_to_memory(
        checkpoint_dir.joinpath("training_data")
    )
    validation_data = file_util.load_to_memory(
        checkpoint_dir.joinpath("validation_data")
    )
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
    del training_data
    del validation_data
    print("Saving model")
    torch.save(model.state_dict(), model_path)
  else:
    print("Loading Model")
    model.load_state_dict(torch.load(model_path))

  test_data = file_util.load_to_memory(
      checkpoint_dir.joinpath("training_data")
  )
  print("Evaluation")
  evaluate_model.evaluate_multiclass_model(
      model=model,
      device=device,
      batch_size=config.sys.batch_size,
      data=[x.dense_data for x in test_data],
      labels=[x.label for x in test_data],
      class_names=IDX2LABEL,
  )


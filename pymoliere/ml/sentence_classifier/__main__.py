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
from typing import List
import numpy as np
import pickle
import torch


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


def load_training_data_from_ckpt(ckpt_dir:Path)->List[Record]:
  res = []
  for path in tqdm(file_util.get_part_files(ckpt_dir)):
    with open(path, 'rb') as f:
      part = pickle.load(f)
    for record in tqdm(part):
      if record["sent_type"] in LABEL2IDX:
        res.append(TrainingData(
          dense_data=torch.FloatTensor(
            np.append(
              record["embedding"],
              record["sent_idx"] / float(record["sent_total"])
            )
          ),
          label=LABEL2IDX[record["sent_type"]],
          date=record["date"],
        ))
  return res


if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere sentence_classifier with the following parameters:")
  print(config)

  sent_w_emb_ckpt_dir = (
      Path(config.shared_scratch)
      .joinpath("dask_checkpoints")
      .joinpath("sentences_with_embedding")  # name from construct.__main__
  )
  if not file_util.is_result_saved(sent_w_emb_ckpt_dir):
    print("Error, failed to find `sentences_with_embedding` checkpoint")
    print("Run pymoliere.construct with:")
    print("`--stop_after_ckpt sentences_with_embedding`")
    exit(1)

  print("Loading training data")
  data = load_training_data_from_ckpt(sent_w_emb_ckpt_dir)

  print("Sorting by date")
  data.sort(key=lambda x: x.date)

  print("Splitting training / test")
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
  print(len(data))
  print(f"{len(training_data)} - {len(validation_data)} - {len(test_data)}")
  print("Last Training Date:", training_data[-1].date)
  print("Last Validation Date:", validation_data[-1].date)
  print("Last Test Date:", test_data[-1].date)

  print("Prepping model")
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda:0")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier()
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


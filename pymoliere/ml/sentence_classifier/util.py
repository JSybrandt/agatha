import torch
from pymoliere.util.misc_util import Record
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple, Any, Iterable

# NamedTuples did not play well with distributed pickles
class TrainingData():
  def __init__(self, dense_data, label, date):
    self.dense_data=dense_data
    self.label=label
    self.date=date

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

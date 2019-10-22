BERT_EMB_DIM = 768

from pathlib import Path
from pymoliere.construct import file_util
from typing import Iterable, Any
from random import random
from tqdm import tqdm

def load_random_sample(
    data_dir:Path,
    value_sample_rate:float=1,
    partition_sample_rate:float=1
)->Iterable[Any]:
  assert value_sample_rate > 0
  assert value_sample_rate <= 1
  assert partition_sample_rate > 0
  assert partition_sample_rate <= 1
  assert file_util.is_result_saved(data_dir)
  for part in tqdm(file_util.get_part_files(data_dir)):
    if random() < partition_sample_rate:
      for rec in file_util.load_part(part):
        if random() < value_sample_rate:
          yield rec

from typing import List, Tuple, Any, Optional, Dict, Callable, Iterable
import torch
import torch
from pytorch_transformers import (
    BertModel,
    BertTokenizer,
)
import dask.bag as dbag
import dask.dataframe as ddf
import numpy as np
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

def record_to_vector(
    records:dbag.Bag,
    text_field:str,
    id_field:str,
    embedding_field:str,
    scibert_data_dir:Path,
    bert_batch_size:int,
    bert_use_gpu:bool,
)->dbag.Bag:
  """
  Converts a collection of records to an indexed dataframe.  Resulting
  dataframe will have one vector per record, created by using scibert on the
  text_field.
  """
  return records.filter(
      lambda r: text_field in r and len(r[text_field]) > 0
  ).map_partitions(
      _record_part_to_vectors,
      text_field=text_field,
      id_field=id_field,
      embedding_field=embedding_field,
      scibert_data_dir=scibert_data_dir,
      bert_batch_size=bert_batch_size,
      bert_use_gpu=bert_use_gpu,
  )

def _record_part_to_vectors(
    records:Iterable[Tuple[str, str]],
    text_field:str,
    id_field:str,
    embedding_field:str,
    scibert_data_dir:Path,
    bert_batch_size:int,
)->Iterable[Dict[str, Any]]:
  records = list(records)
  print("Setting up SciBert")
  tok = BertTokenizer.from_pretrained(scibert_data_dir)
  model = BertModel.from_pretrained(scibert_data_dir)

  for start_idx in range(0, len(records), bert_batch_size):
    end_idx = min(len(records), start_idx+bert_batch_size)
    texts = pad_sequence(
      sequences = [
        torch.tensor(tok.encode(r[text_field], add_special_tokens=True))
        for r in records[start_idx:end_idx]
      ],
      batch_first=True,
    )
    embedding = model(texts)[-1].detach().numpy()
    for rec, emb in zip(records[start_idx:end_idx], embedding):
      yield {id_field: rec[id_field], embedding_field: emb}

def embed_texts(
    texts:List[str],
    scibert_data_dir:Path,
    bert_batch_size:int,
)->Iterable[np.ndarray]:
  "A lower-level function to get text embeddings without the bulk of records"
  tok = BertTokenizer.from_pretrained(scibert_data_dir)
  model = BertModel.from_pretrained(scibert_data_dir)
  for start_idx in range(0, len(records), bert_batch_size):
    end_idx = min(len(records), start_idx+bert_batch_size)
    texts = pad_sequence(
      sequences = [
        torch.tensor(tok.encode(r[text_field], add_special_tokens=True))
        for r in records[start_idx:end_idx]
      ],
      batch_first=True,
    )
    embedding = model(texts)[-1].detach().numpy()
    yield embedding


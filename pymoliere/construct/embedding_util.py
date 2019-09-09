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
    scibert_data_dir:Path,
    id_fn:Callable[[Dict[str,Any]], str],
    bert_batch_size:int,
    bert_use_gpu:bool,
)->ddf.DataFrame:
  """
  Converts a collection of records to an indexed dataframe.  Resulting
  dataframe will have one vector per record, created by using scibert on the
  text_field.
  """
  meta = pd.DataFrame(columns=['id', 'embedding'])
  meta.id.astype(str)
  meta.embedding.astype(np.array([1,2,3], dtype=np.float32))

  return records.filter(
      lambda r: text_field in r and len(r[text_field]) > 0
  ).map(
      lambda r: (id_fn(r), r[text_field])
  ).map_partitions(
      _record_part_to_vectors,
      scibert_data_dir=scibert_data_dir,
      bert_batch_size=bert_batch_size,
      bert_use_gpu=bert_use_gpu,
  ).to_dataframe(
      meta=meta
  ).reset_index(
      drop=True
  )


def _record_part_to_vectors(
    records:Iterable[Tuple[str, str]],
    scibert_data_dir:Path,
    bert_batch_size:int,
    bert_use_gpu:int
)->List[Dict[str, Any]]:
  assert not bert_use_gpu
  records = list(records)
  print("Setting up SciBert")
  tok = BertTokenizer.from_pretrained(scibert_data_dir)
  model = BertModel.from_pretrained(scibert_data_dir)

  if bert_use_gpu:
    model = model.cuda()

  ids = [r[0] for r in records]
  embeddings = []
  num_batches = int(np.ceil(len(records)/float(bert_batch_size)))
  for start_idx in range(0, len(records), num_batches):
    end_idx = min(len(records), start_idx+bert_batch_size)

    texts = pad_sequence(
      sequences = [
        torch.tensor(tok.encode(b[1], add_special_tokens=True))
        for b in records[start_idx:end_idx]
      ],
      batch_first=True,
    )
    if bert_use_gpu:
      texts = texts.cuda()
    embedding = model(texts)[-1].detach().numpy()
    for emb in embedding:
      embeddings.append(emb.tolist())

  return pd.DataFrame({"id":ids, "embedding": embeddings})

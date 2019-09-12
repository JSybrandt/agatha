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
from pymoliere.util.misc_util import iter_to_batches
from tqdm import tqdm

def record_to_vector(
    records:dbag.Bag,
    text_field:str,
    id_field:str,
    embedding_field:str,
    scibert_data_dir:Path,
    batch_size:int,
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
      batch_size=batch_size,
      bert_use_gpu=bert_use_gpu,
  )

def _record_part_to_vectors(
    records:Iterable[Tuple[str, str]],
    text_field:str,
    id_field:str,
    embedding_field:str,
    scibert_data_dir:Path,
    batch_size:int,
)->Iterable[Dict[str, Any]]:
  records = list(records)
  print("Setting up SciBert")
  tok = BertTokenizer.from_pretrained(scibert_data_dir)
  model = BertModel.from_pretrained(scibert_data_dir)

  for start_idx in range(0, len(records), batch_size):
    end_idx = min(len(records), start_idx+batch_size)
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
    batch_size:int,
    use_gpu:bool,
)->Iterable[np.ndarray]:
  "A lower-level function to get text embeddings without the bulk of records"
  "use_gpu uses it if available"

  use_gpu = use_gpu and torch.cuda.is_available()

  if not hasattr(embed_texts, "tok"):
    print("configuring tok")
    embed_texts.tok = BertTokenizer.from_pretrained(scibert_data_dir)
  if not hasattr(embed_texts, "model"):
    print("configuring model")
    embed_texts.model = BertModel.from_pretrained(scibert_data_dir)
    if use_gpu:
      print("Sending to GPU...")
      embed_texts.model = embed_texts.model.cuda()
    embed_texts.model.eval()

  tok = embed_texts.tok
  model = embed_texts.model

  for batch in tqdm(iter_to_batches(texts, batch_size)):
    try:
      sequs = pad_sequence(
        sequences=[
          torch.tensor(tok.encode(t))
          for t in batch
        ],
        batch_first=True,
      )
      with torch.no_grad():
        embedding = model(sequs)[-1].detach().numpy()
      yield embedding
    except:
      # Switch to per-text embedding to find the bad one
      for t in batch:
        try:
          with torch.no_grad():
            yield model(
                torch.tensor([tok.encode(t)])
            )[-1].detach().numpy()
        except:
          print(t)

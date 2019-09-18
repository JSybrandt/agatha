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
import logging
from dask.distributed import Lock
from pymoliere.util.misc_util import Record

GLOBAL_MODEL = {
    "embed_texts_tok": None,
    "embed_texts_model": None,
    "embed_texts_dev": None,
    "embed_texts": None
}

def init_model(scibert_data_dir:Path, disable_gpu:bool):
  lock = Lock(f"init_model")
  while(not lock.acquire(timeout=5)):
    pass
  # THREADSAFE
  if GLOBAL_MODEL["embed_texts"] is None:
    logging.info("Configuring Scibert Model in embed_texts")
    if torch.cuda.is_available() and not disable_gpu:
      dev = torch.device("cuda")
    else:
      dev = torch.device("cpu")
    tok = BertTokenizer.from_pretrained(scibert_data_dir)
    model = BertModel.from_pretrained(scibert_data_dir)
    model.eval()
    # if torch.cuda.device_count() > 1:
      # model = torch.nn.DataParallel(model)
    model = model.to(dev)
    GLOBAL_MODEL["embed_texts_tok"] = tok
    GLOBAL_MODEL["embed_texts_model"] = model
    GLOBAL_MODEL["embed_texts_dev"] = dev
    GLOBAL_MODEL["embed_texts"] = True
  else:
    raise Exception("Ran twice on same machine!!!")
  # End Threadsafe
  lock.release()


def embed_texts(
    texts:List[str],
    batch_size:int,
)->Iterable[np.ndarray]:
  "A lower-level function to get text embeddings without the bulk of records"
  "use_gpu uses it if available"
  tok = GLOBAL_MODEL["embed_texts_tok"]
  model = GLOBAL_MODEL["embed_texts_model"]
  dev = GLOBAL_MODEL["embed_texts_dev"]
  assert tok is not None
  assert model is not None
  assert dev is not None

  for batch in tqdm(iter_to_batches(texts, batch_size)):
    sequs = pad_sequence(
      sequences=[
        torch.tensor(tok.encode(t))
        for t in batch
      ],
      batch_first=True,
    ).to(dev)
    with torch.no_grad():
      yield model(sequs)[-1].cpu().detach().numpy()

def embed_records(
    records:Iterable[Record],
    batch_size:int,
    text_field:str,
    id_field:str="id",
    out_embedding_field:str="embedding",
)->Iterable[Record]:
  """
  Analogous to embed_texts, but for a partition of records.
  Each record must contain text_field and id_field.
  Result are truncated records containing only id_field and out_embedding_field.
  """
  tok = GLOBAL_MODEL["embed_texts_tok"]
  model = GLOBAL_MODEL["embed_texts_model"]
  dev = GLOBAL_MODEL["embed_texts_dev"]
  assert tok is not None
  assert model is not None
  assert dev is not None

  res = []
  for batch in tqdm(iter_to_batches(records, batch_size)):
    texts = list(map(lambda x: x[text_field], batch))
    ids = list(map(lambda x: x[id_field], batch))
    sequs = pad_sequence(
      sequences=[
        torch.tensor(tok.encode(t))
        for t in texts
      ],
      batch_first=True,
    ).to(dev)
    with torch.no_grad():
      embs = model(sequs)[-1].cpu().detach().numpy()
    for _id, emb in zip(ids, embs):
      res.append({
        id_field: _id,
        out_embedding_field: emb,
      })
  return res


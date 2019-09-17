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

GLOBAL_MODEL = {
    "embed_texts_tok": None,
    "embed_texts_model": None,
    "embed_texts_dev": None,
    "embed_texts": None
}

def init_model(scibert_data_dir:Path):
  lock = Lock(f"init_model")
  while(not lock.acquire(timeout=5)):
    pass
  # THREADSAFE
  if GLOBAL_MODEL["embed_texts"] is None:
    logging.info("Configuring Scibert Model in embed_texts")
    dev = torch.device('cuda') if torch.cuda.is_available() else "cpu"
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

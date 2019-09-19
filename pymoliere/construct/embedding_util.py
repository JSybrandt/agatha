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
from pymoliere.util.misc_util import iter_to_batches
import logging
from dask.distributed import Lock
from pymoliere.util.misc_util import Record
from pymoliere.construct import dask_process_global as dpg

def get_scibert_initializer(
    scibert_data_dir:Path,
    disable_gpu:bool,
)->Tuple[str, dpg.Initializer]:
  def _init():
    if torch.cuda.is_available() and not disable_gpu:
      dev = torch.device("cuda")
    else:
      dev = torch.device("cpu")
    tok = BertTokenizer.from_pretrained(scibert_data_dir)
    model = BertModel.from_pretrained(scibert_data_dir)
    model.eval()
    model.to(dev)
    return (dev, tok, model)
  return "embedding_util:dev,tok,model", _init


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

  dev, tok, model = dpg.get("embedding_util:dev,tok,model")

  res = []
  for batch in iter_to_batches(records, batch_size):
    texts = list(map(lambda x: x[text_field], batch))
    ids = list(map(lambda x: x[id_field], batch))
    sequs = pad_sequence(
      sequences=[
        torch.tensor(tok.encode(t))
        for t in texts
      ],
      batch_first=True,
    ).to(dev)
    try:
      with torch.no_grad():
        embs = model(sequs)[-1].cpu().detach().numpy()
      for _id, emb in zip(ids, embs):
        res.append({
          id_field: _id,
          out_embedding_field: emb,
        })
    except:
      import pickle
      from pymoliere.construct.file_util import touch_random_unused_file
      out_file = touch_random_unused_file(
          Path("/zfs/safrolab/users/jsybran/pymoliere/data/err_reports"),
          ".err.pkl",
      )
      with open(out_file, "wb") as f:
        f.write(pickle.dumps(batch))
      print("Encountered Error")
  return res


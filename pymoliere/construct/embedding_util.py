from dask.distributed import Lock
from pathlib import Path
from pymoliere.construct import dask_process_global as dpg
from pymoliere.ml.sentence_classifier import (
    record_to_sentence_classifier_input,
    sentence_classifier_output_to_labels,
)
from pymoliere.util.misc_util import Record
from pymoliere.util.misc_util import iter_to_batches
from pytorch_transformers import (
    BertModel,
    BertTokenizer,
)
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from typing import Tuple, Iterable, ClassVar
import dask.bag as dbag
import dask.dataframe as ddf
import logging
import math
import numpy as np
import torch

def get_pytorch_device_initalizer(
    disable_gpu:bool,
)->Tuple[str, dpg.Initializer]:
  def _init():
    if torch.cuda.is_available() and not disable_gpu:
      return torch.device("cuda")
    else:
      return torch.device("cpu")
  return "embedding_util:device", _init

def get_scibert_initializer(
    scibert_data_dir:Path,
)->Tuple[str, dpg.Initializer]:
  def _init():
    device = dpg.get("embedding_util:device")
    tok = BertTokenizer.from_pretrained(scibert_data_dir)
    model = BertModel.from_pretrained(scibert_data_dir)
    model.eval()
    model.to(device)
    return (tok, model)
  return "embedding_util:tok,model", _init


def get_pretrained_model_initializer(
  name:str,
  model_class:ClassVar,
  data_dir:Path,
  **model_kwargs
)->Tuple[str, dpg.Initializer]:
  def _init():
    device = dpg.get("embedding_util:device")
    model = model_class(**model_kwargs)
    model.load_state_dict(
        torch.load(
          str(data_dir),
          map_location=device,
        )
    )
    model.eval()
    return model
  return f"embedding_util:{name}", _init


def apply_sentence_classifier_to_part(
    records:Iterable[Record],
    batch_size:int,
    sentence_classifier_name="sentence_classifier",
    predicted_type_suffix=":pred",
    sentence_type_field="sent_type",
)->Iterable[Record]:
  device = dpg.get("embedding_util:device")
  model = dpg.get(f"embedding_util:{sentence_classifier_name}")

  res = []
  for rec_batch in iter_to_batches(records, batch_size):
    model_input = torch.stack(
        [record_to_sentence_classifier_input(r) for r in rec_batch]
    ).to(device)
    predicted_labels = sentence_classifier_output_to_labels(model(model_input))
    for r, lbl in zip(rec_batch, predicted_labels):
      r[sentence_type_field] = lbl+predicted_type_suffix
      res.append(r)
  print(len(res))
  return res


def embed_records(
    records:Iterable[Record],
    batch_size:int,
    text_field:str,
    max_sequence_length:int,
    out_embedding_field:str="embedding",
    show_pbar:bool=False,
)->Iterable[Record]:
  """
  Introduces an embedding field to each record, indicated the scibert embedding
  of the supplied text field.
  """

  dev = dpg.get("embedding_util:device")
  tok, model = dpg.get("embedding_util:tok,model")

  res = []
  # PBar is necessary when using embed_records in helper scripts.
  pbar = tqdm(
      iter_to_batches(records, batch_size),
      total=int(len(records) / batch_size),
      disable=not show_pbar,
  )
  for batch in pbar:
    texts = list(map(lambda x: x[text_field], batch))
    sequs = pad_sequence(
      sequences=[
        torch.tensor(tok.encode(t)[:max_sequence_length])
        for t in texts
      ],
      batch_first=True,
    ).to(dev)
    with torch.no_grad():
      embs = (
          model(sequs)[-2]
          .mean(axis=1)
          .cpu()
          .detach()
          .numpy()
      )
    for record, emb in zip(batch, embs):
      record[out_embedding_field] = emb
      res.append(record)
  return res

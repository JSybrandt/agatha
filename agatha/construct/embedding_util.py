from pathlib import Path
from agatha.construct import dask_process_global as dpg
from agatha.util.misc_util import Record
from agatha.util.misc_util import iter_to_batches
from transformers import (
    BertModel,
    BertTokenizer,
)
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Iterable, ClassVar
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

def get_bert_initializer(
    bert_model:str,
)->Tuple[str, dpg.Initializer]:
  """
  The bert_model may be a path or any provided by the transformers module.
  For instance "bert-base-uncased"
  """
  def _init():
    device = dpg.get("embedding_util:device")
    tok = BertTokenizer.from_pretrained(bert_model)
    model = BertModel.from_pretrained(bert_model)
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


def embed_records(
    records:Iterable[Record],
    batch_size:int,
    text_field:str,
    max_sequence_length:int,
    out_embedding_field:str="embedding",
)->Iterable[Record]:
  """
  Introduces an embedding field to each record, indicated the bert embedding
  of the supplied text field.
  """

  dev = dpg.get("embedding_util:device")
  tok, model = dpg.get("embedding_util:tok,model")

  res = []
  for batch in iter_to_batches(records, batch_size):
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

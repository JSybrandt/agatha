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

def embed_texts(
    texts:List[str],
    scibert_data_dir:Path,
    batch_size:int,
)->Iterable[np.ndarray]:
  "A lower-level function to get text embeddings without the bulk of records"
  "use_gpu uses it if available"
  self = embed_texts

  if not hasattr(embed_texts, "setup"):
    print("Conficuting Scibert Model in embed_texts")
    self.dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    self.tok = BertTokenizer.from_pretrained(scibert_data_dir)
    self.model = BertModel.from_pretrained(scibert_data_dir)
    self.model.eval()
    self.model.to(self.dev)
    self.setup = True

  for batch in tqdm(iter_to_batches(texts, batch_size)):
    sequs = pad_sequence(
      sequences=[
        torch.tensor(self.tok.encode(t), device=self.dev)
        for t in batch
      ],
      batch_first=True,
    )
    with torch.no_grad():
      embedding = self.model(sequs)[-1].cpu().detach().numpy()
    yield embedding

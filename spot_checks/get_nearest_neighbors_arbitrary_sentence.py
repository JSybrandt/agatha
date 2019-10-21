#!/usr/bin/env python3


from pytorch_transformers import (
    BertModel,
    BertTokenizer,
)
import faiss
from pymoliere.construct.file_util import  load_to_memory
from pathlib import Path
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
import redis
import numpy as np

REDIS_SERVER = "nodecugi"
SHARED_SCRATCH_ROOT = Path("/scratch4/jsybran/pymoliere_avg_hidden_emb_scratch")
BERT_ROOT = Path(
    "/zfs/safrolab/users/jsybran/pymoliere/data/scibert_scivocab_uncased"
)

print("Making Redis Connection")
redis_client = redis.Redis(REDIS_SERVER)
redis_client.ping()

print("Loading Bert")
tokenizer = BertTokenizer.from_pretrained(str(BERT_ROOT))
bert_model = BertModel.from_pretrained(str(BERT_ROOT))
bert_model.eval()

print("Loading FAISS Index")
faiss_index = faiss.read_index(
    str(
      SHARED_SCRATCH_ROOT
      .joinpath("faiss_index")
      .joinpath("final.index")
    )
)

print("Reading from stdin until eof.")
for line in sys.stdin:
  line = line.strip().lower()
  sequs = pad_sequence(
    sequences=[
      torch.tensor(tokenizer.encode(line)[:500])
    ],
    batch_first=True,
  )
  embs = bert_model(sequs)[0].mean(axis=1).cpu().detach().numpy()
  _, neighs = faiss_index.search(embs, 30)
  with redis_client.pipeline() as pipe:
    for neigh_idx in neighs[0]:
      pipe.get(np.int64(neigh_idx).tobytes())
    print(pipe.execute())

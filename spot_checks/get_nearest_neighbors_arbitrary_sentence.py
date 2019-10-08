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

# SHARED_SCRATCH_ROOT = Path("/scratch4/jsybran/pymoliere_scratch")
# BERT_ROOT = Path(
    # "/zfs/safrolab/users/jsybran/pymoliere/data/scibert_scivocab_uncased"
# )

# print("Loading Bert")
# tokenizer = BertTokenizer.from_pretrained(str(BERT_ROOT))
# bert_model = BertModel.from_pretrained(str(BERT_ROOT))
# bert_model.eval()

# print("Loading FAISS Index")
# faiss_index = faiss.read_index(
    # str(
      # SHARED_SCRATCH_ROOT
      # .joinpath("faiss_index")
      # .joinpath("final.index")
    # )
# )

# print("Loading inverted index")
# index_data = load_to_memory(
    # SHARED_SCRATCH_ROOT
    # .joinpath("dask_checkpoints")
    # .joinpath("hash_and_graph_key")
# )
# print("Converting to inverted index")
# collisions = 0
# inv_index = {}
# for val in index_data:
  # if val["id"] in inv_index:
    # inv_index[val["id"]].append(val["name"])
    # collisions += 1
  # else:
    # inv_index[val["id"]] = [val["name"]]
# print(f"Found {collisions} collisions within {len(index_data)} total keys.")
# del index_data


print("Reading from stdin until eof.")
for line in sys.stdin:
  line = line.strip().lower()
  sequs = pad_sequence(
    sequences=[
      torch.tensor(tokenizer.encode(line)[:500])
    ],
    batch_first=True,
  )
  embs = bert_model(sequs)[-1].cpu().detach().numpy()
  _, neighs = faiss_index.search(embs, 30)
  for neigh_idx in neighs[0]:
    print(inv_index[neigh_idx])

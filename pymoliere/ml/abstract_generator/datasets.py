import torch
import pickle
from pathlib import Path
from sqlitedict  import SqliteDict
from torch.utils.data import Dataset
from pymoliere.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
import random
from typing import Dict, Tuple, List

def shift_text_features_for_training(
    batch:Dict[str, torch.LongTensor]
)-> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor]]:
  model_in_keys = {'text', 'year', 'mesh'}
  expected_keys = {'text', 'pos', 'dep', 'ent'}
  for key in model_in_keys.union(expected_keys):
    assert key in batch, f"Batch missing {key}"
  expected = {
      # remember, that batch is 2nd dim, and seq is first
      # remove first token
      key: batch[key][1:, :]
      for key in expected_keys
  }
  model_in = {
      key: batch[key]
      for key in model_in_keys
  }
  # Crop the last element from the model input text
  model_in["text"] = model_in["text"][:-1]
  return (model_in, expected)

def collate_encoded_abstracts(
    batch:List[Dict[str,List[int]]]
)->Dict[str, torch.LongTensor]:
  needed_keys = ['text', 'pos', 'dep', 'ent', 'year', 'mesh']
  for needed_key in needed_keys:
    for b in batch:
      assert needed_key in b, f"Encoded abstract missing {needed_key}"
  return {
      key: torch.nn.utils.rnn.pad_sequence(
          [torch.LongTensor(b[key]) for b in batch]
      )
      for key in needed_keys
  }

class EncodedAbstracts(Dataset):
  def __init__(
      self,
      abstract_ds:Dataset,
      tokenizer:AbstractGeneratorTokenizer,
      max_text_length:int,
      max_mesh_length:int,
      title_only=False
  ):
    super(EncodedAbstracts, self).__init__()
    self.abstract_ds = abstract_ds
    self.title_only = title_only
    self.max_text_length = max_text_length
    self.max_mesh_length = max_mesh_length
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.abstract_ds)

  def __getitem__(self, data_idx):
    abstract = self.abstract_ds[data_idx]
    # Asserts
    for needed_key in ['year', 'mesh_headings', 'sentences']:
      assert needed_key in abstract, f"Abstract missing {needed_key}"
    for sent in abstract["sentences"]:
      for needed_key in ["text", "type", "tags", "ents"]:
        assert needed_key in sent, f"Sentence missing {needed_key}"
    # Pick a start sentence
    sents = abstract["sentences"]
    if self.title_only:
      # If we're only outputting titles, just subset to that
      sents = list(filter(lambda s: s["type"]=="title", sents))
      selected_sent_idx = 0
    else:
      selected_sent_idx = random.randint(0, len(sents)-1)
    # encode text starting with the selected sentence to max length
    result = None
    for sent_idx in range(selected_sent_idx, len(sents)):
      is_first = sent_idx==0
      is_last = sent_idx == len(sents)-1
      tmp_tokens = self.tokenizer.encode_sentence(
          sents[sent_idx], is_first, is_last
      )
      assert "text" in tmp_tokens
      if result is None:
        result = tmp_tokens
      else:
        for key in result:
          result[key] += tmp_tokens[key]
      if len(result["text"]) >= self.max_text_length:
        break
    # truncate what we got to max text length
    for key in result:
      result[key] = result[key][:self.max_text_length]
    assert "year" not in result
    result["year"] = [self.tokenizer.encode_year(abstract["year"])]
    assert "mesh" not in result
    result["mesh"] = [
        self.tokenizer.encode_mesh(m)
        for m in abstract["mesh_headings"][:self.max_mesh_length]
    ]
    return result


class KVStoreDictDataset(Dataset):
  def __init__(self, db_dir:Path):
    super(KVStoreDictDataset, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = sorted(list(db_dir.glob("*.sqlite")))
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    self.sizes = [0]
    for p in self.db_paths:
      with SqliteDict(p, flag='r') as conn:
        self.sizes.append(self.sizes[-1] + len(conn))
    self.connections = {}

  def __len__(self):
    return self.sizes[-1]

  def __getitem__(self, data_idx):
    for db_idx in range(len(self.db_paths)):
      if self.sizes[db_idx] <= data_idx < self.sizes[db_idx+1]:
        local_idx = data_idx - self.sizes[db_idx]
        if db_idx not in self.connections:
          self.connections[db_idx] = SqliteDict(self.db_paths[db_idx], flag='r')
        return self.connections[db_idx][str(local_idx)]

class LoadWholeKVStore(Dataset):
  def __init__(self, db_dir):
    super(LoadWholeKVStore, self).__init__()
    assert db_dir.is_dir(), "Must supply a dir"
    self.db_paths = sorted(list(db_dir.glob("*.sqlite")))
    assert len(self.db_paths) > 0, "Dir must contain at least one sqlite file"
    with SqliteDict(self.db_paths[0], flag='r') as conn:
      self.data = list(conn.values())

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx]


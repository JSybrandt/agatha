import torch
from pathlib import Path
from pymoliere.ml.util.embedding_index import EmbeddingIndex
from pymoliere.util.sqlite3_graph import Sqlite3Graph
from pymoliere.util import database_util as dbu
import json
from typing import Optional, List
import random
from itertools import chain

def point_cloud_training_collate(positive_examples:List[torch.FloatTensor]):
  # Each matrix inside batch is #points X dim
  assert len(positive_examples) > 0
  assert all(map(lambda p: len(p.shape)==2, positive_examples))
  emb_dim = positive_examples[0].shape[1]
  assert all(map(lambda p: p.shape[1] == emb_dim, positive_examples))
  # They are all # points X embedding dim
  # Shuffle all embeddings together
  shuffled_embeddings =  torch.cat(positive_examples)
  shuffled_embeddings[
      torch.randperm(shuffled_embeddings.shape[0])
  ] = shuffled_embeddings
  # shuffled_embeddings is total#points X embedding_dim
  # parse out shuffled embeddings into negative training examples
  negative_examples = []
  start_idx = 0
  for pos_ex in positive_examples:
    end_idx = start_idx + pos_ex.shape[0]
    negative_examples.append(shuffled_embeddings[start_idx:end_idx])
    start_idx = end_idx
  # negative_examples is the same shapes as positive_examples
  # Shuffle together a list of positives and negatives
  all_examples = list(chain(
      map(lambda p: (p, 1), positive_examples),
      map(lambda n: (n, 0), negative_examples),
  ))
  random.shuffle(all_examples)
  tensors, labels = zip(*all_examples)
  return {
      "point_clouds": torch.nn.utils.rnn.pad_sequence(tensors),
      "labels": torch.FloatTensor(labels)
  }


class EntityIndex(object):
  def __init__(self, entity_dir:Path, entity_type:str):
    entity_dir = Path(entity_dir)
    assert entity_dir.is_dir()

    self.count_paths = sorted(list(
      entity_dir.glob(f"entity_count_{entity_type}_*.txt")
    ))
    self.name_paths = sorted(list(
      entity_dir.glob(f"entity_names_{entity_type}_*.json")
    ))
    assert len(self.count_paths) > 0, "Missing count files"
    assert len(self.count_paths) == len(self.name_paths), \
        "Unequal # of name/count files"
    self.prefix_count = self._get_prefix_count(self.count_paths)
    self.path_idx2names = {}

  @staticmethod
  def _get_prefix_count(count_paths:List[Path])->List[int]:
    counts = [0]
    for p in count_paths:
      with open(p) as f:
        this_count = int(f.read())
      counts.append(this_count + counts[-1])
    return counts

  def get_path_idx(self, global_idx:int)->int:
    assert global_idx < len(self)
    for path_idx, start_idx in enumerate(self.prefix_count):
      if start_idx <= global_idx:
        return path_idx

  def __len__(self):
    return self.prefix_count[-1]

  def __getitem__(self, global_idx:int)->str:
    path_idx = self.get_path_idx(global_idx)
    assert path_idx is not None
    if path_idx not in self.path_idx2names:
      with open(self.name_paths[path_idx]) as f:
        self.path_idx2names[path_idx] = json.load(f)
    return self.path_idx2names[path_idx][global_idx-self.prefix_count[path_idx]]


class PointCloudDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      embedding_dim:int,
      entity_dir:Path,
      embedding_index:EmbeddingIndex,
      graph_index:Sqlite3Graph,
      source_node_type:str,
      neighbor_cloud_type:str,
  ):
    assert embedding_dim > 0
    self.embedding_dim = embedding_dim
    self.neighbor_cloud_type = neighbor_cloud_type
    assert self.neighbor_cloud_type in {
      dbu.ENTITY_TYPE,
      dbu.LEMMA_TYPE,
      dbu.MESH_TERM_TYPE,
      dbu.NGRAM_TYPE,
      dbu.PREDICATE_TYPE,
    }
    self.embedding_index = embedding_index
    self.graph_index = graph_index
    self.sentence_names = EntityIndex(entity_dir, source_node_type)

  def __len__(self):
    return len(self.sentence_names)

  def __getitem__(self, idx:int)->torch.tensor:
    name = self.sentence_names[idx]
    neighs = [n for n in self.graph_index[name] if n[0] == self.neighbor_cloud_type]
    if len(neighs) > 0:
      return torch.stack([
        torch.FloatTensor(self.embedding_index[n]) for n in neighs
      ])
    else:
      return torch.zeros((1, self.embedding_dim))

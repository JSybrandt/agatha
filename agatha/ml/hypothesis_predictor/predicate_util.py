from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
import numpy as np
from typing import List, Tuple, Set
from agatha.util.entity_types import PREDICATE_TYPE, UMLS_TERM_TYPE
import random
import torch
from dataclasses import dataclass


@dataclass
class PredicateEmbeddings:
  subj:np.array
  obj:np.array
  subj_neigh:List[np.array]
  obj_neigh:List[np.array]


def parse_predicate_name(predicate_name:str)->Tuple[str, str]:
  """
  Converts a name like: p:c1365543:causes:c0260397
  into (m:c1365543, m:c0260397)
  """
  typ, sub, vrb, obj = predicate_name.lower().split(":")
  assert typ == PREDICATE_TYPE
  return f"{UMLS_TERM_TYPE}:{sub}", f"{UMLS_TERM_TYPE}:{obj}"


class PredicateObservationGenerator():
  """
  Converts predicate names to predicate observations
  """
  def __init__(
      self,
      graph:Sqlite3LookupTable,
      embeddings:EmbeddingLookupTable,
      neighbor_sample_rate:int,
  ):
    assert neighbor_sample_rate >= 0
    self.graph = graph
    self.embeddings = embeddings
    self.neighbor_sample_rate = neighbor_sample_rate

  def _sample_neighborhood(self, neigh:Set[str])->List[str]:
    neigh = list(neigh)
    if len(neigh) < self.neighbor_sample_rate:
      return neigh
    else:
      return random.sample(neigh, self.neighbor_sample_rate)

  def _get_pred_neigh_from_diff(
      self,
      subj:str,
      obj:str
  )->Tuple[List[str], List[str]]:
    s = set(filter(lambda k: k[0]==PREDICATE_TYPE, self.graph(subj)))
    o = set(filter(lambda k: k[0]==PREDICATE_TYPE, self.graph(obj)))
    s, o = (s-o, o-s)
    return self._sample_neighborhood(s), self._sample_neighborhood(o)

  def __getitem__(self, predicate:str)->PredicateEmbeddings:
    subj, obj = parse_predicate_name(predicate)
    s_neigh, o_neigh = self._get_pred_neigh_from_diff(subj, obj)
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    s_neigh = [self.embeddings[s] for s in s_neigh]
    o_neigh = [self.embeddings[o] for p in o_neigh]
    return PredicateEmbeddings(
        subj=subj,
        obj=obj,
        subj_neigh=subj_neigh,
        obj_neigh=obj_neigh
    )


class PredicateScrambleObservationGenerator(PredicateObservationGenerator):
  """
  Same as above, but the neighborhood comes from randomly selected predicates
  """
  def __init__(self, predicates:List[str], *args, **kwargs):
    PredicateObservationGenerator.__init__(self, *args, **kwargs)
    self.predicates = predicates

  def __getitem__(self, predicate:str):
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    neighs = [
        self.embeddings[predicate]
        for predicate in
        random.sample(self.predicates, self.neighbor_sample_rate*2)
    ]
    return PredicateEmbeddings(
        subj=subj,
        obj=obj,
        subj_neigh=neighs[:neighbor_sample_rate],
        obj_neigh=neighs[neighbor_sample_rate:]
    )


class NegativePredicateGenerator():
  def __init__(self, coded_terms:List[str]):
    self.coded_terms = coded_terms

  def generate(self):
    subj = random.choice(self.coded_terms)
    obj = random.choice(self.coded_terms)
    assert subj[0] == UMLS_TERM_TYPE
    assert obj[0] == UMLS_TERM_TYPE
    # Convert m:c00, m:c11 to p:c00:verb:c11
    subj = subj[2:]
    obj = obj[2:]
    return "{PREDICATE_TYPE}:{subj}:neg:{obj}"


class PredicateBatchGenerator():
  def __init__(
      self,
      graph:Sqlite3LookupTable,
      embeddings:EmbeddingLookupTable,
      predicates:List[str],
      coded_terms:List[str],
      neighbor_sample_rate:int,
      negative_swap_rate:int,
      negative_scramble_rate:int,
  ):
    self.negative_generator = NegativePredicateGenerator(
        coded_terms=coded_terms
    )
    self.scramble_observation_generator = PredicateScrambleObservationGenerator(
        predicates=predicates,
        graph=graph,
        embeddings=embeddings,
        neighbor_sample_rate=neighbor_sample_rate,
    )
    self.observation_generator = PredicateObservationGenerator(
        graph=graph,
        embeddings=embeddings,
        neighbor_sample_rate=neighbor_sample_rate,
    )
    self.negative_swap_rate = negative_swap_rate
    self.negative_scramble_rate  = negative_scramble_rate

  def __call__(self, positive_predicates):
    self.generate(positive_predicates)

  def generate(
      self,
      positive_predicates:List[str]
  )->Tuple[List[PredicateEmbeddings], List[List[PredicateEmbeddings]]]:
    """
    Generates a list of embedding data for each positive predicate
    Generates negative samples associated with each positive predicate

    pos, negs = self.generate(...)
    pos[i] == embeddings related to positive_predicates[i]
    negs[j][i] == embeddings related to the j'th negative sample of i

    collate_predicate_embeddings(pos) == positive model input
    collate_predicate_embeddings(negs[j]) ==
      one of the corresponding negative inputs
    """

    pos = [self.observation_generator[p] for p in positive_predicates]
    negs = []
    for _ in range(self.negative_swap_rate):
      negs.append([
        self.observation_generator[
          self.negative_generator.generate()
        ]
        for _ in positive_predicates
      ])
    for _ in range(self.negative_scramble_rate):
      negs.append([
        self.scramble_observation_generator[
          self.negative_generator.generate()
        ]
        for _ in positive_predicates
      ])
    return pos, negs


def collate_predicate_embeddings(
    predicate_embeddings:PredicateEmbeddings
)->torch.FloatTensor:
  """
  Combine a lost of predicate embeddings stored as multiple np arrays
  into a single pytorch tensor
  if n = len(predicate_embeddings)
  r = neighbor_sample_rate
  d = embedding dimensionality
  dimensions: (2+2(r)) X n X d
  """
  return torch.cat([
    # Subject Embeddings
    torch.FloatTensor([p.subj for p in predicate_embeddings]).unsqueeze(0),
    # Object Embeddings
    torch.FloatTensor([p.obj for p in predicate_embeddings]).unsqueeze(0),
    # Neighbor Embeddings
    torch.nn.utils.rnn.pad_sequence([
      torch.FloatTensor(p.subj_neigh + p.obj_neigh)
      for p in predicate_embeddings
    ])
  ])

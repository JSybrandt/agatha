from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
import numpy as np
from typing import List, Tuple, Set
from agatha.util.entity_types import (
    PREDICATE_TYPE,
    UMLS_TERM_TYPE,
    is_predicate_type,
    is_umls_term_type,
)
import random
import torch
from dataclasses import dataclass


@dataclass
class PredicateEmbeddings:
  subj:np.array
  obj:np.array
  subj_neigh:List[np.array]
  obj_neigh:List[np.array]


def clean_coded_term(term:str)->str:
  """
  If term is not formatted as an agatha coded term key, produces a coded term
  key. Otherwise, just returns the term.
  """
  if is_umls_term_type(term):
    return term.lower()
  else:
    return f"{UMLS_TERM_TYPE}:{term}".lower()


def is_valid_predicate_name(predicate_name:str)->bool:
  if not is_predicate_type(predicate_name):
    return False
  try:
    typ, sub, vrb, obj = predicate_name.lower().split(":")
  except Exception:
    return False
  return (len(sub) > 0) and (len(obj) > 0)


def parse_predicate_name(predicate_name:str)->Tuple[str, str]:
  """Parses subject and object from predicate name strings.

  Predicate names are formatted strings that follow this convention:
  p:{subj}:{verb}:{obj}. This function extracts the subject and object and
  returns coded-term names in the form: m:{entity}. Will raise an exception if
  the predicate name is improperly formatted.

  Args:
    predicate_name: Predicate name in form p:{subj}:{verb}:{obj}.

  Returns:
    The subject and object formulated as coded-term names.

  """
  assert is_predicate_type(predicate_name), \
      f"Not a predicate name: {predicate_name}"
  typ, sub, vrb, obj = predicate_name.lower().split(":")
  assert typ == PREDICATE_TYPE
  return clean_coded_term(sub), clean_coded_term(obj)


def to_predicate_name(
    subj:str,
    obj:str,
    verb:str="unknown",
    )-> str:
  """Converts two names into a predicate of form p:t1:verb:t2

  Assumes that terms are correct Agatha graph keys. This means that we expect
  input terms in the form of m:____. Allows for a custom verb type, but
  defaults to unknown. Output will always be set to lowercase.

  Example usage:

  ```
  to_predicate_name(m:c1, m:c2)
  > p:c1:unknown:c2
  to_predicate_name(m:c1, m:c2, "treats")
  > p:c1:treats:c2
  to_predicate_name(m:c1, m:c2, "TREATS")
  > p:c1:treats:c2
  ```

  Args:
    subj: Subject term. In the form of "m:_____"
    obj: Object term. In the form of "m:_____"
    verb: Optional verb term for resulting predicate.

  Returns:
    Properly formatted predicate containing subject and object. Verb type will
    be set to "UNKNOWN"

  """
  assert is_umls_term_type(subj), \
    f"Called to_predicate_name with bad subject: {subj})"
  assert is_umls_term_type(obj), \
    f"Called to_predicate_name with bad object: {obj})"
  assert ":" not in verb, "Verb cannot contain colon character"
  subj = subj[2:]
  obj = obj[2:]
  return f"{PREDICATE_TYPE}:{subj}:{verb}:{obj}".lower()



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
    assert subj in self.graph, f"Failed to find {subj} in graph."
    assert obj in self.graph, f"Failed to find {obj} in graph."
    s = set(filter(is_predicate_type, self.graph[subj]))
    o = set(filter(is_predicate_type, self.graph[obj]))
    s, o = (s-o, o-s)
    return self._sample_neighborhood(s), self._sample_neighborhood(o)

  def __getitem__(self, predicate:str)->PredicateEmbeddings:
    try:
      subj, obj = parse_predicate_name(predicate)
    except Exception:
      raise Exception(f"Failed to parse predicate: {predicate}")
    subj_neigh, obj_neigh = self._get_pred_neigh_from_diff(subj, obj)
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    subj_neigh = [self.embeddings[s] for s in subj_neigh]
    obj_neigh = [self.embeddings[o] for o in obj_neigh]
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
    subj, obj = parse_predicate_name(predicate)
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
        subj_neigh=neighs[:self.neighbor_sample_rate],
        obj_neigh=neighs[self.neighbor_sample_rate:]
    )




class NegativePredicateGenerator():
  def __init__(
      self,
      coded_terms:List[str],
      graph:Sqlite3LookupTable,
  ):
    "Generates coded terms that appear in graph."
    self.coded_terms = coded_terms
    self.graph = graph
    self._log = []

  def _choose_term(self):
    term = random.choice(self.coded_terms)
    while term not in self.graph:
      term = random.choice(self.coded_terms)
    return term

  def generate(self):
    subj = self._choose_term()
    obj = self._choose_term()
    predicate = to_predicate_name(subj, obj)
    if self._log is not None:
      self._log.append(predicate)
    return predicate

  def disable_debug_log(self):
    self._log = None

  def get_debug_log(self)->List[str]:
    "Returns the last negative predicates since the log was cleared"
    return self._log[:]

  def clear_debug_log(self)->None:
    self._log = []



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
      verbose:bool=False,
  ):
    self.verbose = verbose
    self.negative_generator = NegativePredicateGenerator(
        coded_terms=coded_terms,
        graph=graph,
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

  def get_last_batch_neg_predicates(self)->List[str]:
    log = self.negative_generator.get_debug_log()
    self.negative_generator.clear_debug_log()
    return log

  def __call__(self, positive_predicates):
    return self.generate(positive_predicates)

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

    if self.verbose:
      print("Generating Positives...")
    pos = [self.observation_generator[p] for p in positive_predicates]
    negs = []
    if self.verbose:
      print("Generating Negative Swaps...")
    self.negative_generator.clear_debug_log()
    for _ in range(self.negative_swap_rate):
      negs.append([
        self.observation_generator[
          self.negative_generator.generate()
        ]
        for _ in positive_predicates
      ])
    if self.verbose:
      print("Generating Negative Scrambles...")
    for _ in range(self.negative_scramble_rate):
      negs.append([
        self.scramble_observation_generator[
          self.negative_generator.generate()
        ]
        for _ in positive_predicates
      ])
    return pos, negs


def collate_predicate_embeddings(
    predicate_embeddings:List[PredicateEmbeddings]
)->torch.FloatTensor:
  """Combines a list of predicate embeddings into a single tensor.

  if n = len(predicate_embeddings) r = neighbor_sample_rate d = embedding
  dimensionality, then the result is of size: (2+2(r)) X n X d

  """
  return torch.cat([
    torch.nn.utils.rnn.pad_sequence([
      torch.FloatTensor([p.subj, p.obj] + p.subj_neigh + p.obj_neigh)
      for p in predicate_embeddings
    ])
  ])

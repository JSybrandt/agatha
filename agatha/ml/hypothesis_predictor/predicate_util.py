from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
import numpy as np
from typing import List, Tuple, Set, Dict, Any
from agatha.util.entity_types import (
    PREDICATE_TYPE,
    UMLS_TERM_TYPE,
    is_predicate_type,
    is_umls_term_type,
)
import random
import torch
from dataclasses import dataclass
import time


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
    start = time.time()
    subj_neigh, obj_neigh = self._get_pred_neigh_from_diff(subj, obj)
    subj = self.embeddings[subj]
    obj = self.embeddings[obj]
    subj_neigh = [self.embeddings[s] for s in subj_neigh]
    obj_neigh = [self.embeddings[o] for o in obj_neigh]
    end = time.time()
    #print("Generating PredicateEmbeddings:", int(end-start))
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

  def _choose_term(self):
    term = random.choice(self.coded_terms)
    while term not in self.graph:
      term = random.choice(self.coded_terms)
    return term

  def generate(self):
    subj = self._choose_term()
    obj = self._choose_term()
    predicate = to_predicate_name(subj, obj)
    return predicate


class PredicateExampleDataset(torch.utils.data.Dataset):
  def __init__(
      self,
      predicate_ds:torch.utils.data.Dataset,
      all_predicates:List[str],
      graph:Sqlite3LookupTable,
      embeddings:EmbeddingLookupTable,
      coded_terms:List[str],
      neighbor_sample_rate:int,
      negative_swap_rate:int,
      negative_scramble_rate:int,
      preload_on_first_call:bool=True,
      verbose:bool=False,
  ):
    self.graph = graph
    self.embeddings = embeddings
    self.verbose = verbose
    self.predicate_ds = predicate_ds
    self.negative_generator = NegativePredicateGenerator(
        coded_terms=coded_terms,
        graph=graph,
    )
    self.scramble_observation_generator = PredicateScrambleObservationGenerator(
        predicates=all_predicates,
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
    self._first_call = preload_on_first_call

  def __len__(self)->int:
    return len(self.predicate_ds)

  def __getitem__(self, idx:int)->Dict[str, Any]:
    if self._first_call:
      print("Worker preloading...")
      start = time.time()
      self.graph.preload()
      self.embeddings.preload()
      end = time.time()
      print(f"Worker preloading: {int(end-start)}s")
      self._first_call = False
    start = time.time()
    positive_predicate = self.predicate_ds[idx]
    positive_observation = self.observation_generator[positive_predicate]
    negative_predicates = []
    negative_observations = []
    for _ in range(self.negative_swap_rate):
      p = self.negative_generator.generate()
      negative_predicates.append(p)
      negative_observations.append(self.observation_generator[p])
    for _ in range(self.negative_swap_rate):
      p = self.negative_generator.generate()
      negative_predicates.append(p)
      negative_observations.append(self.scramble_observation_generator[p])
    end = time.time()
    #print(f"Worker produced batch: {int(end-start)}")
    return dict(
        positive_predicate=positive_predicate,
        positive_observation=positive_observation,
        negative_predicates=negative_predicates,
        negative_observations=negative_observations,
    )



def collate_predicate_embeddings(
    predicate_embeddings:List[PredicateEmbeddings]
):
  return torch.cat([
    torch.nn.utils.rnn.pad_sequence([
      torch.FloatTensor([p.subj, p.obj] + p.subj_neigh + p.obj_neigh)
      for p in predicate_embeddings
    ])
  ])

def collate_predicate_training_examples(
    examples:List[Dict[str,Any]],
)->Dict[str, Any]:
  """
  Takes a list of results from PredicateExampleDataset and produces tensors
  for input into the agatha training model.
  """
  positive_predicates = [e["positive_predicate"] for e in examples]
  positive_observations = collate_predicate_embeddings(
      [e["positive_observation"] for e in examples]
  )
  negative_predicates_list = \
      list(zip(*[e["negative_predicates"] for e in examples]))
  negative_observations_list = [
      collate_predicate_embeddings(neg_obs)
      for neg_obs in zip(*[e["negative_observations"] for e in examples])
  ]
  return dict(
      positive_predicates=positive_predicates,
      positive_observations=positive_observations,
      negative_predicates_list=negative_predicates_list,
      negative_observations_list=negative_observations_list,
  )


from pymoliere.ml.util.entity_index import EntityIndex
from pymoliere.ml.util.embedding_index import (
    PreloadedEmbeddingIndex, EmbeddingIndex
)
from pymoliere.util import database_util as dbu
from pathlib import Path
import torch
from typing import Dict, Tuple, Any, List, Set
import random
from copy import deepcopy
from itertools import chain
from pymoliere.util.sqlite3_graph import Sqlite3Graph
from dataclasses import dataclass
import numpy as np

@dataclass
class HypothesisTensors:
  subject_embedding:torch.FloatTensor
  object_embedding:torch.FloatTensor
  subject_neighbor_embeddings:torch.FloatTensor
  object_neighbor_embeddings:torch.FloatTensor
  label:torch.FloatTensor

@dataclass
class PredicateObservation:
  subject_name:str
  object_name:str
  subject_embedding:np.array
  object_embedding:np.array
  subject_neighbor_embeddings:List[np.array]
  object_neighbor_embeddings:List[np.array]
  label:int

IDX2VERB = [
    "administered_to", "affects", "associated_with", "augments", "causes",
    "coexists_with", "compared_with", "complicates", "converts_to",
    "diagnoses", "disrupts", "higher_than", "inhibits", "interacts_with",
    "isa", "location_of", "lower_than", "manifestation_of", "measurement_of",
    "measures", "method_of", "neg_administered_to", "neg_affects",
    "neg_associated_with", "neg_augments", "neg_causes", "neg_coexists_with",
    "neg_complicates", "neg_converts_to", "neg_diagnoses", "neg_disrupts",
    "neg_higher_than", "neg_inhibits", "neg_interacts_with", "neg_isa",
    "neg_location_of", "neg_lower_than", "neg_manifestation_of",
    "neg_measurement_of", "neg_measures", "neg_method_of", "neg_occurs_in",
    "neg_part_of", "neg_precedes", "neg_predisposes", "neg_prevents",
    "neg_process_of", "neg_produces", "neg_same_as", "neg_stimulates",
    "neg_treats", "neg_uses", "occurs_in", "part_of", "precedes",
    "predisposes", "prevents", "process_of", "produces", "same_as",
    "stimulates", "treats", "uses", "UNKNOWN", "INVALID"
]
VERB2IDX = {v:i for i, v in enumerate(IDX2VERB)}

def _sample_relevant_neighbors(
    term:str,
    excluded_term:str,
    neighbors_per_term:int,
    graph_index:Sqlite3Graph,
)->List[str]:
  items = [
      n for n in graph_index[term]
      if excluded_term not in PredicateLoader.parse_predicate_name(n)
  ]
  if len(items) <= neighbors_per_term:
    return items
  else:
    return random.sample(items, neighbors_per_term)


def generate_predicate_observation(
    subj:str,
    obj:str,
    neighbors_per_term:int,
    graph_index:Sqlite3Graph,
    embedding_index:EmbeddingIndex,
    label:int=1,
):
    subj_neigh = _sample_relevant_neighbors(
        subj, obj, neighbors_per_term, graph_index
    )
    obj_neigh = _sample_relevant_neighbors(
        obj, subj, neighbors_per_term, graph_index
    )
    return PredicateObservation(
        subject_name=subj,
        object_name=obj,
        subject_embedding=embedding_index[subj],
        object_embedding=embedding_index[obj],
        subject_neighbor_embeddings=[
          embedding_index[n] for n in subj_neigh
        ],
        object_neighbor_embeddings=[
          embedding_index[n] for n in obj_neigh
        ],
        label=label
    )

class PredicateLoader(torch.utils.data.Dataset):
  def __init__(
      self,
      embedding_index:EmbeddingIndex,
      graph_index:Sqlite3Graph,
      entity_dir:Path,
      neighbors_per_term:int,
  ):
    self.predicate_index = EntityIndex(
        entity_dir, entity_type=dbu.PREDICATE_TYPE
    )
    self.embedding_index = embedding_index
    self.graph_index = graph_index
    self.neighbors_per_term = neighbors_per_term

  @staticmethod
  def parse_predicate_name(predicate_name:str)->Tuple[str, str, str]:
    components = predicate_name.split(":")
    assert len(components) == 4
    assert components[0] == dbu.PREDICATE_TYPE
    return components[1:]

  def __len__(self):
    return len(self.predicate_index)

  def __getitem__(self, idx:int)->PredicateObservation:
    predicate = self.predicate_index[idx]
    subj, _, obj = self.parse_predicate_name(predicate)
    subj = f"{dbu.MESH_TERM_TYPE}:{subj}"
    obj = f"{dbu.MESH_TERM_TYPE}:{obj}"
    return generate_predicate_observation(
        subj, obj, self.neighbors_per_term, self.graph_index,
        self.embedding_index, 1
    )


class TestPredicateLoader(torch.utils.data.Dataset):
  def __init__(
      self,
      test_data_dir:Path,
      embedding_index:EmbeddingIndex,
      graph_index:Sqlite3Graph,
      neighbors_per_term:int,
  ):
    self.neighbors_per_term=neighbors_per_term
    self.embedding_index = embedding_index
    self.graph_index = graph_index
    published_path = Path(test_data_dir).joinpath("published.txt")
    noise_path = Path(test_data_dir).joinpath("noise.txt")
    assert published_path.is_file()
    assert noise_path.is_file()
    self.subjs_objs_labels = []
    num_failures = 0
    for path in [published_path, noise_path]:
      with open(path) as pred_file:
        for line in pred_file:
          subj, obj, year = line.lower().strip().split("|")
          if subj in graph_index and obj in graph_index:
            label = 1 if int(year) > 0 else 0
            self.subjs_objs_labels[(subj, obj, label)]

  def __len__(self):
    return len(self.subjs_objs_labels)

  def __getitem__(self, idx:int)->PredicateObservation:
    subj, obj, label = self.subjs_objs_labels[idx]
    subj = f"{dbu.MESH_TERM_TYPE}:{subj}"
    obj = f"{dbu.MESH_TERM_TYPE}:{obj}"
    return generate_predicate_observation(
        subj, obj, self.neighbors_per_term, self.graph_index,
        self.embedding_index, 1
    )


def observations_to_tensors(samples:List[PredicateObservation])->HypothesisTensors:
  return HypothesisTensors(
      subject_embedding=torch.FloatTensor([
        s.subject_embedding for s in samples
      ]),
      object_embedding=torch.FloatTensor([
        s.object_embedding for s in samples
      ]),
      subject_neighbor_embeddings=torch.nn.utils.rnn.pad_sequence([
        torch.FloatTensor(s.subject_neighbor_embeddings)
        for s in samples
      ]),
      object_neighbor_embeddings=torch.nn.utils.rnn.pad_sequence([
        torch.FloatTensor(s.object_neighbor_embeddings)
        for s in samples
      ]),
      label=torch.FloatTensor([
        s.label for s in samples
      ]),
  )

def generate_negative_scramble_batch(
    positive_samples:List[PredicateObservation],
    neighbors_per_term:int,
)->HypothesisTensors:
  negative_samples = []
  # Record all neighbors
  all_neighbors = list(chain.from_iterable(map(
      lambda x: chain(
        x.subject_neighbor_embeddings,
        x.object_neighbor_embeddings,
      ),
      positive_samples
  )))
  # Record all subj-obj
  all_entities = list(chain.from_iterable(map(
      lambda x: [x.subject_embedding, x.object_embedding],
      positive_samples
  )))
  # Create a negative sample for each positive
  for _ in positive_samples:
    negative_samples.append(PredicateObservation(
      subject_name=None,  # these fields will not be used in the to-tensors
      object_name=None,
      subject_embedding=random.choice(all_entities),
      object_embedding=random.choice(all_entities),
      subject_neighbor_embeddings=[
        random.choice(all_neighbors)
        for _ in range(random.randint(1, neighbors_per_term))
      ],
      object_neighbor_embeddings=[
        random.choice(all_neighbors)
        for _ in range(random.randint(1, neighbors_per_term))
      ],
      label=0,
    ))
  return observations_to_tensors(negative_samples)

def neighbor_entities(predicates:List[str])->Set[str]:
  res = set()
  for pred in predicates:
    s, _, o = PredicateLoader.parse_predicate_name(pred)
    res.add(s)
    res.add(o)
  return res

def generate_negative_swap_batch(
    positive_samples:List[PredicateObservation],
    neighbors_per_term:int,
    graph_index:Sqlite3Graph,
    embedding_index:EmbeddingIndex,
)->HypothesisTensors:
  negative_samples = []
  all_entities = set()
  for s in positive_samples:
    all_entities.add(s.subject_name)
    all_entities.add(s.object_name)
  all_entities = list(all_entities)

  for _ in positive_samples:
    subject_name = random.choice(all_entities)
    # Note that subj_name is in invalid_partners
    invalid_partnerns = neighbor_entities(graph_index[subject_name])
    object_name = random.choice(all_entities)
    while object_name in invalid_partnerns:
      object_name = random.choice(all_entities)
    subj_neigh = _sample_relevant_neighbors(
        subject_name, object_name, neighbors_per_term, graph_index
    )
    obj_neigh = _sample_relevant_neighbors(
        object_name, subject_name, neighbors_per_term, graph_index
    )
    negative_samples.append(PredicateObservation(
      subject_name=subject_name,
      object_name=object_name,
      subject_embedding=embedding_index[subject_name],
      object_embedding=embedding_index[object_name],
      subject_neighbor_embeddings=[
        embedding_index[n] for n in subj_neigh
      ],
      object_neighbor_embeddings=[
        embedding_index[n] for n in obj_neigh
      ],
      label=0,
    ))
  return observations_to_tensors(negative_samples)

def predicate_collate(
    positive_samples:List[PredicateObservation],
    neg_scrambles_per:int,
    neg_swaps_per:int,
    neighbors_per_term:int,
    graph_index:Sqlite3Graph,
    embedding_index:EmbeddingIndex,
)->List[HypothesisTensors]:
  """
  Outputs a list of comparisons. The FIRST element of the list is the
  positive class, then all following are of equal length, but of the
  negative class. We're going to compare the ranking between the first and
  all others.
  """
  assert neg_swaps_per + neg_scrambles_per > 0, "Must set some negative samples"
  res  = [observations_to_tensors(positive_samples)]
  if neg_scrambles_per > 0:
    res += [
      generate_negative_scramble_batch(positive_samples, neighbors_per_term)
      for _ in range(neg_scrambles_per)
    ]
  if neg_swaps_per > 0:
    res += [
      generate_negative_swap_batch(
        positive_samples,
        neighbors_per_term,
        graph_index,
        embedding_index
      ) for _ in range(neg_swaps_per)
    ]
  return res

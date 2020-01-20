from pymoliere.ml.util.entity_index import EntityIndex
from pymoliere.ml.util.embedding_index import (
    PreloadedEmbeddingIndex, EmbeddingIndex
)
from pymoliere.util import database_util as dbu
from pathlib import Path
import torch
from typing import Dict, Tuple, Any, List
import random
from copy import deepcopy
from itertools import chain

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

class PredicateLoader(torch.utils.data.Dataset):
  def __init__(
      self,
      entity_dir:Path,
      embedding_dir:Path,
  ):
    self.entity_index = EntityIndex(entity_dir, entity_type=dbu.PREDICATE_TYPE)
    self.embedding_index = PreloadedEmbeddingIndex(
        embedding_dir=embedding_dir,
        entity_dir=entity_dir,
        select_entity_type=dbu.MESH_TERM_TYPE,
    )

  @staticmethod
  def parse_predicate_name(predicate_name:str)->Tuple[str, str, str]:
    components = predicate_name.split(":")
    assert len(components) == 4
    assert components[0] == dbu.PREDICATE_TYPE
    return components[1:]

  def __len__(self):
    return len(self.entity_index)

  def __getitem__(self, idx:int)->Dict[str, Any]:
    subj, verb, obj = self.parse_predicate_name(self.entity_index[idx])
    subj = f"{dbu.MESH_TERM_TYPE}:{subj}"
    obj = f"{dbu.MESH_TERM_TYPE}:{obj}"
    return dict(
        subj_name=subj,
        subj_emb=self.embedding_index[subj],
        obj_name=obj,
        obj_emb=self.embedding_index[obj],
        verb_name=verb,
    )

class TestPredicateLoader(torch.utils.data.Dataset):
  def __init__(
      self,
      test_data_dir:Path,
      embedding_index:EmbeddingIndex
  ):
    published_path = test_data_dir.joinpath("published.txt")
    noise_path = test_data_dir.joinpath("noise.txt")
    assert published_path.is_file()
    assert noise_path.is_file()
    self.predicates = []
    num_failures = 0
    for path in [published_path, noise_path]:
      with open(path) as pred_file:
        for line in pred_file:
          subj, obj, year = line.lower().strip().split("|")
          subj = f"{dbu.MESH_TERM_TYPE}:{subj}"
          obj = f"{dbu.MESH_TERM_TYPE}:{obj}"
          if subj in embedding_index and obj in embedding_index:
            self.predicates.append(dict(
              subj_name=subj,
              subj_emb=embedding_index[subj],
              obj_name=obj,
              obj_emb=embedding_index[obj],
              verb_name="UNKNOWN" if path == published_path else "INVALID",
            ))
          else:
            print("FAILED:", line.strip())
            num_failures += 1
    if num_failures > 0:
      print("Failed:", num_failures)
      print("Succeeded:", len(self.predicates))

  def __len__(self):
    return len(self.predicates)

  def __getitem__(self, idx):
    return self.predicates[idx]

def predicate_collate(
    predicate_data:List[Dict[str, Any]],
    num_negative_samples:int,
    name2idx:Dict[str, Any],
)->Dict[str, Any]:
  def gen_neg_sample():
    subj = random.choice(predicate_data)
    obj = random.choice(predicate_data)
    while obj == subj:
      obj = random.choice(predicate_data)
    return dict(
        subj_name=subj["subj_name"],
        subj_emb=subj["subj_emb"],
        obj_name=subj["obj_name"],
        obj_emb=subj["obj_emb"],
        verb_name="INVALID"
    )
  def safe_get(name, name2idx):
    name_or_none = name2idx.get(name)
    if name_or_none is None:
      return name2idx["UNKNOWN"]
    return name_or_none

  subj_emb = []
  obj_emb = []
  verb_idx = []
  subj_idx = []
  obj_idx = []
  labels = []
  neg_samples = [gen_neg_sample() for _ in range(num_negative_samples)]
  for pred in chain(predicate_data, neg_samples):
    subj_idx.append(safe_get(pred["subj_name"], name2idx))
    obj_idx.append(safe_get(pred["obj_name"], name2idx))
    subj_emb.append(pred["subj_emb"])
    obj_emb.append(pred["obj_emb"])
    safe_verb_idx = safe_get(pred["verb_name"], VERB2IDX)
    verb_idx.append(safe_verb_idx)
    label = 0 if safe_verb_idx == VERB2IDX["INVALID"] else 1
    labels.append(label)

  assert len(subj_emb) == len(obj_emb) == len(verb_idx) == len(labels)
  return dict(
      # batch
      subj_idx=torch.LongTensor(subj_idx),
      # batch X emb_dim
      subj_emb=torch.FloatTensor(subj_emb),
      # batch
      obj_idx=torch.LongTensor(obj_idx),
      # batch X emb_dim
      obj_emb=torch.FloatTensor(obj_emb),
      # batch
      verbs=torch.LongTensor(verb_idx),
      labels=torch.FloatTensor(labels),
  )

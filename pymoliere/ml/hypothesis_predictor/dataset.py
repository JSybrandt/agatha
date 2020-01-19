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
)->Dict[str, Any]:
  def generate_negative_sample():
    return dict(
        subj_emb=random.choice(predicate_data)["subj_emb"],
        obj_emb=random.choice(predicate_data)["obj_emb"],
    )
  def safe_get_verb(pred):
    verb_or_none = VERB2IDX.get(pred["verb_name"])
    if verb_or_none is None:
      return VERB2IDX["UNKNOWN"]
    return verb_or_none

  subjects = []
  objects = []
  verbs = []
  labels = []
  for pred in predicate_data:
    # Positive example
    subjects.append(pred["subj_emb"])
    objects.append(pred["obj_emb"])
    safe_verb_idx = safe_get_verb(pred)
    verbs.append(safe_verb_idx)
    label = 0 if safe_verb_idx == VERB2IDX["INVALID"] else 1
    labels.append(label)
  for _ in range(num_negative_samples):
    # Negative example
    pred = generate_negative_sample()
    subjects.append(pred["subj_emb"])
    objects.append(pred["obj_emb"])
    verbs.append(VERB2IDX["INVALID"])
    labels.append(0)


  assert len(subjects) == len(objects) == len(verbs) == len(labels)
  return dict(
      # batch X emb_dim
      subjects=torch.FloatTensor(subjects),
      # batch X emb_dim
      objects=torch.FloatTensor(objects),
      # batch
      verbs=torch.LongTensor(verbs),
      labels=torch.FloatTensor(labels),
  )

def train_predicate_collate(predicate_data):
  return predicate_collate(predicate_data, len(predicate_data))

def test_predicate_collate(predicate_data):
  return predicate_collate(predicate_data, 0)

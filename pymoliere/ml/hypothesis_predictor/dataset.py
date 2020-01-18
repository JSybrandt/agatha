from pymoliere.ml.util.entity_index import EntityIndex
from pymoliere.ml.util.embedding_index import PreloadedEmbeddingIndex
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
    "stimulates", "treats", "uses", "UNKNOWN"
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

def predicate_collate(predicate_data:List[Dict[str, Any]])->Dict[str, Any]:
  subjects = []
  objects = []
  verbs = []
  labels = []
  for pred in predicate_data:
    subjects.append(pred["subj_emb"])
    objects.append(pred["obj_emb"])
    verb_or_none = VERB2IDX.get(pred["verb_name"])
    if verb_or_none is None:
      verb_or_none = VERB2IDX["UNKNOWN"]
    verbs.append(verb_or_none)
  labels = ([1] * len(subjects)) + ([0] * len(subjects))
  subjects = subjects + subjects
  # Negative set has the objects rotated by 1
  objects = objects + [objects[-1]] + objects[:-1]
  verbs = verbs + verbs
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


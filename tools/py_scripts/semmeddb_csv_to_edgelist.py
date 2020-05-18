#!/usr/bin/env python3
from fire import Fire
from pathlib import Path
from agatha.util import semmeddb_util as sm
from agatha.util import entity_types as typs
from typing import Iterable
import json

def get_sentence_key(predicate:sm.Predicate)->str:
  # version 1
  return "{TYP}:{pmid}:1:{sent}".format(
      TYP=typs.SENTENCE_TYPE,
      pmid=predicate["pmid"],
      sent=predicate["sent_idx"],
  ).lower()

def get_mesh_key(predicate:sm.Predicate, id_field:str)->str:
  return "{TYP}:{val}".format(
      TYP=typs.MESH_TERM_TYPE,
      val=predicate[id_field],
  ).lower()

def get_text_keys(predicate:sm.Predicate, name_field:str)->Iterable[str]:
  name = predicate[name_field].lower().replace(" ", "_")
  yield f"e:{name}"
  if "_" in name:
    yield f"n:{name}"
  else:
    yield f"l:noun:{name}"

def get_adjacent_names(predicate:sm.Predicate)->Iterable[str]:
  yield get_sentence_key(predicate)
  yield get_mesh_key(predicate, "subj_ids")
  yield get_mesh_key(predicate, "obj_ids")
  yield from get_text_keys(predicate, "subj_names")
  yield from get_text_keys(predicate, "obj_names")

def main(
  semmeddb_csv_path:Path,
  output_json_path:Path,
  cut_date:sm.Datestr=None,
  edge_weight:float=1.0,
):
  semmeddb_csv_path = Path(semmeddb_csv_path)
  output_json_path = Path(output_json_path)
  assert semmeddb_csv_path.is_file()
  assert not output_json_path.exists()

  predicates = sm.parse(semmeddb_csv_path)
  if cut_date is not None:
    predicates = sm.filter_by_date(predicates, cut_date)

  with open(output_json_path, 'w') as json_file:
    for predicate in predicates:
      pred_key = sm.predicate_to_key(predicate)
      for neigh in get_adjacent_names(predicate):
        json_file.write(json.dumps({"key": pred_key, "value": neigh}))
        json_file.write("\n")
        json_file.write(json.dumps({"key": neigh, "value": pred_key}))
        json_file.write("\n")



if __name__=="__main__":
  Fire(main)

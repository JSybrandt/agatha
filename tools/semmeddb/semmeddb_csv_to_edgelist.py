#!/usr/bin/env python3
from fire import Fire
from pathlib import Path
from pymoliere.util import semmeddb_util as sm
from pymoliere.util import database_util as db_util
from typing import Iterable

def get_sentence_key(predicate:sm.Predicate)->str:
  # version 1
  return "{TYP}:{pmid}:1:{sent}".format(
      TYP=db_util.SENTENCE_TYPE,
      pmid=predicate["pmid"],
      sent=predicate["sent_idx"],
  ).lower()

def get_mesh_key(predicate:sm.Predicate, id_field:str)->str:
  return "{TYP}:{val}".format(
      TYP=db_util.MESH_TERM_TYPE,
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
  output_tsv_path:Path,
  cut_date:sm.Datestr=None,
  edge_weight:float=1.0,
):
  semmeddb_csv_path = Path(semmeddb_csv_path)
  output_tsv_path = Path(output_tsv_path)
  assert semmeddb_csv_path.is_file()
  assert not output_tsv_path.exists()

  predicates = sm.parse(semmeddb_csv_path)
  if cut_date is not None:
    predicates = sm.filter_by_date(predicates, cut_date)

  with open(output_tsv_path, 'w') as tsv_file:
    for predicate in predicates:
      pred_key = sm.predicate_to_key(predicate)
      for neigh in get_adjacent_names(predicate):
        tsv_file.write(f"{pred_key}\t{neigh}\t{edge_weight}\n")



if __name__=="__main__":
  Fire(main)

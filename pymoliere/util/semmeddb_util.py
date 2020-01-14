from datetime import datetime
from csv import DictReader
from pathlib import Path
from typing import Iterable, Dict
from collections import defaultdict
from tqdm import tqdm
from copy import copy
from itertools import product

# Used to distinguish a reg string from %Y-%m-%d
Datestr = str
# Used to distinguish a reg string from {subj}\t{typ}\t{obj}
PredicateKey = str
Predicate = Dict[str, str]

def assert_datestr(date:Datestr)->None:
  datetime.strptime(date, "%Y-%m-%d")

def split_multi_term_predicate(predicate:Predicate)->Iterable[Predicate]:
  term_fields = ["subj", "obj"]
  element_lists = [[], []]
  # for subj and obj
  for term_idx, term in enumerate(term_fields):
    # Group all splits in ids and names
    element_lists[term_idx] += zip(
      *[predicate[f"{term}{suff}"].split("|") for suff in ["_ids", "_names"]]
    )
  # Get the cross product of subjects and objects
  for (subj_ids, subj_names), (obj_ids, obj_names) in product(*element_lists):
    res = copy(predicate)
    res["subj_ids"] = subj_ids
    res["subj_names"] = subj_names
    res["obj_ids"] = obj_ids
    res["obj_names"] = obj_names
    yield res


def parse(
    semmeddb_csv_path:Path,
    silent_tqdm:bool=False,
)->Iterable[Predicate]:
  semmeddb_csv_path = Path(semmeddb_csv_path)
  with open(semmeddb_csv_path) as f:
    for predicate in tqdm(DictReader(f), disable=silent_tqdm):
      yield from split_multi_term_predicate(predicate)


def filter_by_date(
    semmeddb:Iterable[Predicate],
    cut_date:Datestr,
)->Iterable[Predicate]:
  assert_datestr(cut_date)
  return filter(
      lambda r: r["date"] < cut_date,
      semmeddb
  )


def predicate_to_key(p:Predicate)->str:
  return f"{p['subj_ids']}\t{p['pred_type']}\t{p['obj_ids']}"


def earliest_occurances(
    semmeddb:Iterable[Predicate],
)->Dict[PredicateKey, Datestr]:
  # Earliest invalid date
  pred2date = defaultdict(lambda: "0000-00-00")
  for predicate in semmeddb:
    key = predicate_to_key(predicate)
    if pred2date[key] > predicate["date"]:
      pred2date[key] = predicate["date"]
  return pred2date






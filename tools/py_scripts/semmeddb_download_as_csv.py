#!/usr/bin/env python3
from fire import Fire
from datetime import datetime
import pymysql
from pymysql.cursors import SSDictCursor
from tqdm import tqdm
from typing import Dict, Any, Iterable, List, Tuple
from pathlib import Path
from csv import DictWriter

# READ: https://skr3.nlm.nih.gov/SemMedDB/dbinfo.html

QUERY="""
  SELECT
      c.dp AS date,
      p.pmid AS pmid,
      s.number AS sent_idx,
      s.type AS ti_or_ab,
      p.predicate AS pred_type,
      p.subject_cui AS subj_ids,
      p.subject_name AS subj_names,
      a.subject_text AS subj_text,
      p.subject_semtype AS subj_type,
      p.object_cui AS obj_ids,
      p.object_name AS obj_names,
      a.object_text AS obj_text,
      p.object_semtype AS obj_type
  FROM
    PREDICATION AS p
      JOIN SENTENCE AS s ON p.sentence_id = s.sentence_id
      JOIN CITATIONS AS c ON s.pmid = c.pmid
      join PREDICATION_AUX AS a ON p.predication_id = a.predication_id
  {LIMIT_STMT}
  ;
"""
FIELD_NAMES = [
  'date', 'pmid', 'sent_idx', 'ti_or_ab', 'pred_type', 'subj_ids',
  'subj_names', 'subj_text', 'subj_type', 'obj_ids', 'obj_names', 'obj_text',
  'obj_type',
]

DATE_FORMAT = "%Y-%m-%d"

def clean_date(date:str):
  "Dates are given in YYYY {Month Text} {Day Idx}"
  tokens = date.strip().split()
  # Has the effect of cleaning any extra spaces
  date = " ".join(tokens)
  try:
    if len(tokens) == 3: # Y M D
      # Parse year, plaintext month abbrev. and day
      return (
          datetime
          .strptime(date, "%Y %b %d")
          .strftime(DATE_FORMAT)
      )
    elif len(tokens) == 2: # Y M
      # Parse year, plaintext month abbrev.
      return (
          datetime
          .strptime(date, "%Y %b")
          .strftime(DATE_FORMAT)
      )
    else: # Either we only have the year, or something weird happened
      return datetime(int(tokens[0]), 1, 1).strftime(DATE_FORMAT)
  except Exception:
    return datetime(9999, 1, 1).strftime(DATE_FORMAT)

def download_semmeddb(
    output_path:Path,
    database_address:str="jcloud",
    database_name:str="semmeddb_40R",
    database_user:str=None,
    database_password:str=None,
    limit:int=None,
):
  output_path = Path(output_path)
  assert not output_path.exists(), "Refusing to overwrite."
  if limit is None:
    limit = ""
  else:
    limit = f"LIMIT {limit}"
  query = QUERY.format(LIMIT_STMT=limit)
  db_conn = pymysql.connect(
      host=database_address,
      db=database_name,
      user=database_user,
      password=database_password,
      cursorclass=pymysql.cursors.DictCursor,
      charset="utf8",
  )
  with db_conn.cursor(SSDictCursor) as mysql_cursor:
    print("Running:")
    print(query)
    mysql_cursor.execute(query)
    with open(output_path, 'w', newline='') as csv_file:
      writer = DictWriter(csv_file, fieldnames=FIELD_NAMES)
      writer.writeheader()
      for record in tqdm(mysql_cursor):
        record["date"] = clean_date(record["date"])
        writer.writerow(record)

if __name__=="__main__":
  Fire(download_semmeddb)

import pymysql
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from tqdm import tqdm
from typing import Dict, Any, Iterable, List, Tuple
from pymoliere.util.misc_util import Record
from pymoliere.util.db_key_util import (
    SENTENCE_TYPE,
    UMLS_TERM_TYPE,
    GRAPH_TYPE,
    PREDICATE_TYPE,
    GENE_TYPE,
)
import redis

# READ: https://skr3.nlm.nih.gov/SemMedDB/dbinfo.html

QUERY="""
  SELECT
      p.pmid AS pmid,
      s.type AS type,
      s.number AS sent_idx,
      p.subject_cui AS subject,
      p.object_cui AS object,
      p.predicate AS verb
  FROM
    PREDICATION AS p
    JOIN SENTENCE AS s
    ON (p.sentence_id = s.sentence_id);
"""

def get_terms(term_field:str)->List[str]:
  res = []
  for term in term_field.split("|"):
    if term[0].isalpha():
      res.append(f"{term[0]}:{term[1:]}")
    else:
      res.append(f"{GENE_TYPE}:{term}")
  return res

def get_sentence_graph_key(row:Record)->str:
  version = "0"  # for now, we're only going to link to v1 papers
  sent_idx = row["sent_idx"]  # they index abstracts from 1
  if row["type"] == "ti":  # title
    sent_idx = 0 # we index titles from 0
  return ":".join([
      GRAPH_TYPE,
      SENTENCE_TYPE,
      row["pmid"],
      version,
      str(sent_idx)
  ]).lower()


def get_term_graph_keys(row:Record)->List[str]:
  terms = get_terms(row['subject']) + get_terms(row['object'])
  return [
      ":".join([
        GRAPH_TYPE,
        term,
      ]).lower()
      for term in terms
  ]


def get_canonical_term(terms:List[str])->str:
  """
    Picks the best code from the list of codes. If umls term in list, we use
    that. Otherwise we use the lexographically smallest gene.
  """
  smallest_term = None
  for term in terms:
    if term[0] == UMLS_TERM_TYPE:
      return term
    else:
      if smallest_term is None or term < smallest_term:
        smallest_term = term
  return smallest_term


def get_predicate_graph_key(row:Record)->str:
  return ":".join([
    GRAPH_TYPE,
    PREDICATE_TYPE,
    get_canonical_term(get_terms(row["subject"])),
    row["verb"],
    get_canonical_term(get_terms(row["object"])),
  ]).lower()

def to_edges(
    row:Record,
    sent_weight:float,
    term_weight:float
)->Dict[str, Dict[str, int]]:
  # output written in redis ordered set format, as in {source:{target: weight}}
  pred_key = get_predicate_graph_key(row)
  sent_key = get_sentence_graph_key(row)
  terms = get_term_graph_keys(row)
  # result with sentence
  result = {
      pred_key: {
        sent_key: sent_weight,
      },
      sent_key: {pred_key: sent_weight},
  }
  # add terms
  for term in set(terms):
    result[pred_key][term] = term_weight
    result[term] = {pred_key: term_weight}
  return result


if __name__=="__main__":
  config = cpb.SemMedDBAddonConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere build with the following custom parameters:")
  print(config)

  assert config.semmeddb.HasField("address")
  assert config.semmeddb.HasField("db")

  print(f"Connecting to {config.semmeddb.address} : {config.semmeddb.db}")
  semmeddb_connection = pymysql.connect(
      host=config.semmeddb.address,
      db=config.semmeddb.db,
      user=(
        config.semmeddb.user
        if config.semmeddb.HasField("user")
        else None
      ),
      password=(
        config.semmeddb.password
        if config.semmeddb.HasField("password")
        else None
      ),
      cursorclass=pymysql.cursors.DictCursor,
      charset="utf8",
  )

  print(
      f"Connecting to {config.pymolieredb.address}, "
      f"db: {config.pymolieredb.db_num}"
  )
  redis_client = redis.Redis(
      host=config.pymolieredb.address,
      port=config.pymolieredb.port,
      db=config.pymolieredb.db_num,
  )
  try:
    redis_client.ping()
  except:
    raise Exception(f"No redis server running at {config.db.address}")


  with semmeddb_connection.cursor() as mysql_cursor:
    mysql_cursor.execute(QUERY)
    with redis_client.pipeline() as redis_pipeline:
      for row in tqdm(mysql_cursor.fetchall()):
        edge_data = to_edges(
            row=row,
            sent_weight=config.sentence_predicate_weight,
            term_weight=config.term_predicate_weight,
        )
        for source, neighbors in edge_data.items():
          redis_pipeline.zadd(source, neighbors)
        redis_pipeline.execute()
        print(edge_data)

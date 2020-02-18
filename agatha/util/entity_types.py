DATA_BANK_TYPE="d"
ENTITY_TYPE="e"
GENE_TYPE="a"
LEMMA_TYPE="l"
MESH_TERM_TYPE="m"
NGRAM_TYPE="n"
PREDICATE_TYPE="p"
SENTENCE_TYPE="s"
UMLS_TERM_TYPE="m"  # these are now m:c###

ALL_KEYS = set([
  DATA_BANK_TYPE,
  ENTITY_TYPE,
  GENE_TYPE,
  LEMMA_TYPE,
  MESH_TERM_TYPE,
  NGRAM_TYPE,
  PREDICATE_TYPE,
  SENTENCE_TYPE,
  UMLS_TERM_TYPE,
])

def is_graph_key(name:str)->bool:
  if name != name.lower():
    return False
  if len(name) <= 2:
    return False
  if name[1] != ":":
    return False
  return name[0] in ALL_KEYS

def to_graph_key(name:str, key:str)->str:
  assert key in ALL_KEYS
  if is_graph_key(name):
    if name[0] != key:
      raise ValueError(
          f"Attempting to convert a graph key of type {name[0]} to {key}."
      )
    return name
  else:
    return f"{key}:{name.lower()}"

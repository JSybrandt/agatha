from functools import partial

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

# All keys must be one character long
assert all(map(lambda k: len(k)==1, ALL_KEYS)), "INVALID TYPE KEY"

def is_type(type_key:str, name:str)->bool:
  """True if name is an appropriately formatted key of the specified type.

  Names should be in the form "{type_key}:{name}"

  Args:
    name: Unsure name we're querying
    type_key: Single character type, such as one of the strings in this module.

  """
  assert type_key in ALL_KEYS, \
      f"type_key ({type_key}) must be one of: {ALL_KEYS}"
  return len(name) > 2 and name[0] == type_key and name[1] == ":"

# Define type-specific functions
is_data_bank_type = partial(is_type, DATA_BANK_TYPE)
is_entity_type = partial(is_type, ENTITY_TYPE)
is_gene_type = partial(is_type, GENE_TYPE)
is_lemma_type = partial(is_type, LEMMA_TYPE)
is_mesh_term_type = partial(is_type, MESH_TERM_TYPE)
is_ngram_type = partial(is_type, NGRAM_TYPE)
is_predicate_type = partial(is_type, PREDICATE_TYPE)
is_sentence_type = partial(is_type, SENTENCE_TYPE)
is_umls_term_type = partial(is_type, UMLS_TERM_TYPE)


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

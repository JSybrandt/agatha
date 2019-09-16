GRAPH_TYPE="g"
SENTENCE_TYPE="s"
LEMMA_TYPE="l"
ENTITY_TYPE="e"
MESH_TERM_TYPE="m"
DATA_BANK_TYPE="d"


def key_is_type(key:str, key_type:str)->bool:
  "Returns true if the primary type of the given key is `key_type`"
  "For instance, g:s:1234:1:1 is primary type GRAPH_TYPE"
  return key[0] == key_type and key[1] == ":"

def key_contains_type(key:str, key_type:str)->bool:
  "Returns true if the type or sub-type of the given key is key_type"
  "For instance g:s:1234:1:1 contains type SENTENCE_TYPE"
  toks = key.split(":")
  return toks[0] == key_type or toks[1] == key_type

def to_graph_key(key:str)->str:
  assert not key_is_type(key, GRAPH_TYPE)
  return f"{GRAPH_TYPE}:{key}"

def from_graph_key(key:str)->str:
  assert key_is_type(key, GRAPH_TYPE)
  return key[len(GRAPH_TYPE)+1:]

def to_edge(source:str, target:str, weight:float=1):
  "an edge is just a special dict."
  if not key_is_type(source, GRAPH_TYPE):
    raise Exception(f"Invalid source key: '{source}'")
  if not key_is_type(target, GRAPH_TYPE):
    raise Exception(f"Invalid target key: '{target}'")
  return {
      "source": source,
      "target": target,
      "weight": weight,
  }

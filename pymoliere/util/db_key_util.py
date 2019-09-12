GRAPH_TYPE="g"
SENTENCE_TYPE="s"
KEYWORD_TYPE="k"
ENTITY_TYPE="e"

def key_is_type(key:str, key_type:str)->bool:
  "Returns true if the primary type of the given key is `key_type`"
  "For instance, g:s:1234:1:1 is primary type GRAPH_TYPE"
  return key.split(":")[0] == key_type

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


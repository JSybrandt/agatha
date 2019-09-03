from pathlib import Path
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from typing import List, Tuple, Any, Optional, Dict
import spacy
from copy import copy

PROCESS_GLOBAL = {
    "nlp": None,
}

def setup_scispacy(
    scispacy_version:str,
    extra_parts:bool=False,
)->Tuple[Any, UmlsEntityLinker]:
  print("Loading scispacy... Might take a bit.")
  nlp = spacy.load(scispacy_version)

  if extra_parts:
    # Add the abbreviation pipe to the spacy pipeline.
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
    # Add UMLS linker to pipeline
    umls_linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(umls_linker)

  return nlp

def split_sentences(
    elem:Dict[str, Any],
    text_fields:List[str],
    nlp:Any=None,
    scispacy_version:str=None,
    sentence_text_field:str="sentence",
    sentence_idx_field:str="sentence_idx",
)->List[Dict[str, Any]]:
  "Replaces text_fields with sentence"
  if nlp is None:
    if PROCESS_GLOBAL["nlp"] is None:
      PROCESS_GLOBAL["nlp"] = setup_scispacy(scispacy_version)
    nlp = PROCESS_GLOBAL["nlp"]
  res = []

  # Get all non-textual data
  non_text_elem = copy(elem)
  for text_field in text_fields:
    non_text_elem.pop(text_field, None)

  assert sentence_text_field not in non_text_elem
  assert sentence_idx_field not in non_text_elem

  sent_idx = 0
  for field in text_fields:
    if elem[field] is not None:
      doc = nlp(elem[field])
      for sent in doc.sents:
        sent_text = "".join([t.string for t in sent])
        new_elem = copy(non_text_elem)
        new_elem[sentence_text_field] = sent_text.strip()
        new_elem[sentence_idx_field] = sent_idx
        res.append(new_elem)
        sent_idx += 1
  return res

def add_entitites(
    elem:Dict[str, Any],
    text_field:str,
    nlp:Any,
    umls_field:str="umls_terms",
    entity_field:str="entities",
)->Dict[str, Any]:
  assert text_field in elem
  assert umls_field not in elem
  assert entity_field not in elem
  elem[umls_field] = []
  elem[entity_field] = []
  doc = nlp(elem[text_field])
  for ent in doc.ents:
    elem[entity_field].append(str(ent))
    if len(ent._.umls_ents) > 0:
      elem[umls_ents].append(ent._.umls_ents[0][0])
    else:
      elem[umls_ents].append(None)
  return elem

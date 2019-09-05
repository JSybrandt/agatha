from pathlib import Path
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from typing import List, Tuple, Any, Optional, Dict
import spacy
from copy import copy
from spacy_pytorch_transformers import (
    PyTT_WordPiecer,
    PyTT_TokenVectorEncoder,
)
from nltk.tokenize import sent_tokenize

GLOBAL_NLP_OBJS = {
    "analyze_sentence": None,
}

def setup_scispacy(
    scispacy_version:str,
    add_scispacy_parts:bool=False,
    add_scibert_parts:bool=False,
    scibert_dir:Path=None,
)->Tuple[Any, UmlsEntityLinker]:
  print("Loading scispacy... Might take a bit.")
  nlp = spacy.load(scispacy_version)
  if add_scispacy_parts:
    print("\t- And UMLS Component")
    nlp.add_pipe(AbbreviationDetector(nlp))
    nlp.add_pipe(UmlsEntityLinker(resolve_abbreviations=True))
  if add_scibert_parts:
    print("\t- And SciBert Component")
    nlp.add_pipe(
        PyTT_WordPiecer.from_pretrained(
          nlp.vocab,
          str(scibert_dir)
        )
    )
    nlp.add_pipe(
        PyTT_TokenVectorEncoder.from_pretrained(
          nlp.vocab,
          str(scibert_dir)
        )
    )
  return nlp

def split_sentences(
    elem:Dict[str, Any],
    text_fields:List[str],
    sentence_text_field:str="sentence",
    sentence_idx_field:str="sentence_idx",
)->List[Dict[str, Any]]:
  "Replaces text_fields with sentence"

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
      for sent_text in sent_tokenize(elem[field]):
        new_elem = copy(non_text_elem)
        new_elem[sentence_text_field] = sent_text
        new_elem[sentence_idx_field] = sent_idx
        res.append(new_elem)
        sent_idx += 1
  return res


def init_analyze_sentence(
    scispacy_version:str=None,
    scibert_dir:Path=None,
)->None:
    if GLOBAL_NLP_OBJS["analyze_sentence"] is None:
      GLOBAL_NLP_OBJS["analyze_sentence"] = setup_scispacy(
          scispacy_version=scispacy_version,
          scibert_dir=scibert_dir,
          add_scibert_parts=True,
          add_scispacy_parts=True,
      )

def analyze_sentence(
    elem:Dict[str, Any],
    text_field:str,
    nlp:Any=None,
    token_field:str="tokens",
    entity_field:str="entities",
    vector_field:str="vector"
)->Dict[str, Any]:
  "Splits tokens into useful components"
  assert text_field in elem
  assert token_field not in elem
  assert entity_field not in elem
  assert vector_field not in elem

  if nlp is None:
    nlp = GLOBAL_NLP_OBJS["analyze_sentence"]

  elem = copy(elem)
  try:
    doc = nlp(elem[text_field])
    elem[vector_field] = doc.vector.tolist()
    elem[entity_field] = [
      {
        "tok_start": ent.start,
        "tok_end": ent.end,
        "cha_start": ent.start_char,
        "cha_end": ent.end_char,
        "umls": ent._.umls_ents[0][0] if len(ent._.umls_ents) > 0 else None,
        "vector": ent.vector.tolist(),
      }
      for ent in doc.ents
    ]
    elem[token_field] = [
        {
          "cha_start": tok.idx,
          "cha_end": tok.idx + len(tok),
          "lemma": tok.lemma_,
          "pos": tok.pos_,
          "tag": tok.tag_,
          "dep": tok.dep_,
          "vector": tok.vector.tolist() if not tok.is_stop else None,
          "stop": tok.is_stop,
        }
        for tok in doc
    ]
  except Exception(e):
    elem["ERR"] = e.msg
    print(e)
  return elem

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
    document_elem:Dict[str, Any],
    text_data_field:str="text_data",
    sentence_prefix:str="sent",
)->List[Dict[str, Any]]:
  """
  Splits a document into its collection of sentences. In order of text field elements, we split sentences and create new elements for the result. All fields from the original document, as well as the text field (minus the actual text itself) are copied over. For instance:

  {
    "status": "Published",
    "umls": ["C123", "C456"],
    "text_fields": [{
        "text": "Title 1",
        "type": "title"
      }, {
        "text": "This is an abstract. This is another sentence.",
        "type": "abstract:raw",
      }]
  }

  becomes:

  [{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "Title 1",
    "sent_type": "title",
    "sent_idx": 0,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is an abstract.",
    "sent_type": "abstract:raw",
    "sent_idx": 1,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is another sentence.",
    "sent_type": "abstract:raw",
    "sent_idx": 2,
    "sent_total": 3,
  }]
  """
  assert text_data_field in document_elem
  sent_text_key = f"{sentence_prefix}_text"
  sent_idx_key = f"{sentence_prefix}_idx"
  sent_total_key = f"{sentence_prefix}_total"

  res = []

  # Get all non-textual data
  doc_non_text_elem = copy(document_elem)
  doc_non_text_elem.pop(text_data_field)

  sent_idx = 0
  for text_data in document_elem[text_data_field]:
    assert "text" in text_data
    # Holds all additional text features per-sentence in this field
    non_text_attr = copy(doc_non_text_elem)
    for key, val in text_data.items():
      if key != "text":
        key = f"{sentence_prefix}_{key}"
        assert key not in non_text_attr
        non_text_attr[key] = val
    for sentence_text in sent_tokenize(text_data["text"]):
      sent_elem = copy(non_text_attr)
      assert sent_text_key not in sent_elem
      sent_elem[sent_text_key] = sentence_text
      assert sent_idx_key not in sent_elem
      sent_elem[sent_idx_key] = sent_idx
      sent_idx += 1
      res.append(sent_elem)

  # set total
  for r in res:
    assert sent_total_key not in r
    r[sent_total_key] = sent_idx
  return res


def init_analyze_sentence(
    scispacy_version:str=None,
    scibert_dir:Path=None,
)->None:
    if GLOBAL_NLP_OBJS["analyze_sentence"] is None:
      GLOBAL_NLP_OBJS["analyze_sentence"] = setup_scispacy(
          scispacy_version=scispacy_version,
          #scibert_dir=scibert_dir,
          #add_scibert_parts=True,
          #add_scispacy_parts=True,
      )

def analyze_sentence(
    sent_elem:Dict[str, Any],
    text_field:str,
    nlp:Any=None,
    token_field:str="tokens",
    entity_field:str="entities",
    #vector_field:str="vector"
)->Dict[str, Any]:
  "Splits tokens into useful components"
  assert text_field in sent_elem
  assert token_field not in sent_elem
  assert entity_field not in sent_elem
  #assert vector_field not in sent_elem

  if nlp is None:
    nlp = GLOBAL_NLP_OBJS["analyze_sentence"]

  sent_elem = copy(sent_elem)
  try:
    doc = nlp(sent_elem[text_field])
    #sent_elem[vector_field] = doc.vector.tolist()
    sent_elem[entity_field] = [
      {
        "tok_start": ent.start,
        "tok_end": ent.end,
        "cha_start": ent.start_char,
        "cha_end": ent.end_char,
        #"umls": ent._.umls_ents[0][0] if len(ent._.umls_ents) > 0 else None,
        #"vector": ent.vector.tolist(),
      }
      for ent in doc.ents
    ]
    sent_elem[token_field] = [
        {
          "cha_start": tok.idx,
          "cha_end": tok.idx + len(tok),
          "lemma": tok.lemma_,
          "pos": tok.pos_,
          "tag": tok.tag_,
          "dep": tok.dep_,
          #"vector": tok.vector.tolist() if not tok.is_stop else None,
          "stop": tok.is_stop,
        }
        for tok in doc
    ]
  except Exception as e:
    sent_elem["ERR"] = str(e)
    print(e)
  return sent_elem

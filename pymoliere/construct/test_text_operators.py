from pymoliere.construct import text_operators
from pymoliere.util import pipeline_operator
import pandas as pd
from pathlib import Path
import re

class TokenStub():
  def __init__(self, val):
    self.string = val

class SpacyTestStub():
  def __call__(self, doc):
    self.sents = [
      sent.split()
      for sent in [
        x.strip()
        for x in re.findall(r'[^\.\?\!]+[\.\?\!]?', doc)
      ]
    ]
    self.sents = [
        [TokenStub(x+" ") for x in s[:-1]] + [TokenStub(s[-1])]
        for s in self.sents
    ]
    return self


def test_split_sentence_operator():
  input_data = pd.DataFrame([
      {
        "id": 123,
        "title": "Document 1",
        "abstract": "This is an abstract. Hopefully it parses well.",
        "status": "Example",
      },
      {
        "id": 456,
        "title": "Document 2",
        "abstract": "Here's another. Hope its good. Maybe it parses well.",
        "status": "Test",
      },
  ])

  expected = pd.DataFrame([
      {
        "id": 123,
        "status": "Example",
        "sentence": "Document 1",
        "sentence_idx": 0,
      },
      {
        "id": 123,
        "status": "Example",
        "sentence": "This is an abstract.",
        "sentence_idx": 1,
      },
      {
        "id": 123,
        "status": "Example",
        "sentence": "Hopefully it parses well.",
        "sentence_idx": 2,
      },
      {
        "id": 456,
        "status": "Test",
        "sentence": "Document 2",
        "sentence_idx": 0,
      },
      {
        "id": 456,
        "status": "Test",
        "sentence": "Here's another.",
        "sentence_idx": 1,
      },
      {
        "id": 456,
        "status": "Test",
        "sentence": "Hope its good.",
        "sentence_idx": 2,
      },
      {
        "id": 456,
        "status": "Test",
        "sentence": "Maybe it parses well.",
        "sentence_idx": 3,
      },
  ])

  actual = text_operators.text_fields_to_sentences(
    dataframe=input_data,
    text_fields=["title", "abstract"],
    sentence_text_field="sentence",
    sentence_idx_field="sentence_idx",
    nlp=SpacyTestStub()
  )
  col_order = ["id", "status", "sentence", "sentence_idx"]
  assert actual[col_order].equals(expected[col_order])


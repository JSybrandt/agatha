from pymoliere.construct import text_operators
from pymoliere.util import pipeline_operator
from pathlib import Path
import re
import dask
from dask.delayed import delayed
import dask.bag as dbag

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

def get_input():
  return dbag.from_sequence(
    seq=[{
        "id": 123,
        "title": "Document 1",
        "abstract": "This is an abstract. Hopefully it parses well.",
        "status": "Example",
      }, {
        "id": 456,
        "title": "Document 2",
        "abstract": "Here's another. Hope its good. Maybe it parses well.",
        "status": "Test",
    }],
    npartitions=1,
  )

def get_expected():
  return [{
        "id": 123,
        "status": "Example",
        "sentence": "Document 1",
        "sentence_idx": 0,
      }, {
        "id": 123,
        "status": "Example",
        "sentence": "This is an abstract.",
        "sentence_idx": 1,
      }, {
        "id": 123,
        "status": "Example",
        "sentence": "Hopefully it parses well.",
        "sentence_idx": 2,
      }, {
        "id": 456,
        "status": "Test",
        "sentence": "Document 2",
        "sentence_idx": 0,
      }, {
        "id": 456,
        "status": "Test",
        "sentence": "Here's another.",
        "sentence_idx": 1,
      }, {
        "id": 456,
        "status": "Test",
        "sentence": "Hope its good.",
        "sentence_idx": 2,
      }, {
        "id": 456,
        "status": "Test",
        "sentence": "Maybe it parses well.",
        "sentence_idx": 3,
  }]


def test_split_sentences_simple():
  input_data = get_input()
  expected = get_expected()
  actual = input_data.map(
      text_operators.split_sentences,
      # --
      text_fields=["title", "abstract"],
      sentence_text_field="sentence",
      sentence_idx_field="sentence_idx",
      nlp=SpacyTestStub()
  ).flatten().compute()
  assert actual == expected

def test_split_sentences_hard():
  input_data = get_input()
  expected = get_expected()
  # Smaller text lang for scipy
  nlp = dask.delayed(text_operators.setup_scispacy)("en_core_web_sm")
  actual = text_operators.SplitSentencesOperator(
      name="test_SplitSentencesOperator",
      input_data=input_data,
      text_fields=["title", "abstract"],
      sentence_text_field="sentence",
      sentence_idx_field="sentence_idx",
      nlp=nlp,
      # --
      skip_scratch=True,
  ).get_value().compute()
  print(actual)
  assert actual == expected

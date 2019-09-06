from pymoliere.construct import text_util
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

def get_documents():
  return dbag.from_sequence(
    seq=[{
        "id": 123,
        "status": "Example",
        "text_data": [{
            "text": "Document 1",
            "type": "title",
          },{
            "text": "This is an abstract. Hopefully it parses well.",
            "type": "abstract:raw",
        }]
      }, {
        "id": 456,
        "status": "Test",
        "text_data": [{
            "text": "Document 2",
            "type": "title",
          },{
            "text": "Here's another. Hope its good. Maybe it parses well.",
            "type": "abstract:raw",
        }]
    }],
    npartitions=1,
  )

def get_sentences():
  return [{
        "id": 123,
        "status": "Example",
        "sent_type": "title",
        "sent_text": "Document 1",
        "sent_idx": 0,
        "sent_total": 3,
      }, {
        "id": 123,
        "status": "Example",
        "sent_type": "abstract:raw",
        "sent_text": "This is an abstract.",
        "sent_idx": 1,
        "sent_total": 3,
      }, {
        "id": 123,
        "status": "Example",
        "sent_type": "abstract:raw",
        "sent_text": "Hopefully it parses well.",
        "sent_idx": 2,
        "sent_total": 3,
      }, {
        "id": 456,
        "status": "Test",
        "sent_type": "title",
        "sent_text": "Document 2",
        "sent_idx": 0,
        "sent_total": 4,
      }, {
        "id": 456,
        "status": "Test",
        "sent_type": "abstract:raw",
        "sent_text": "Here's another.",
        "sent_idx": 1,
        "sent_total": 4,
      }, {
        "id": 456,
        "status": "Test",
        "sent_type": "abstract:raw",
        "sent_text": "Hope its good.",
        "sent_idx": 2,
        "sent_total": 4,
      }, {
        "id": 456,
        "status": "Test",
        "sent_type": "abstract:raw",
        "sent_text": "Maybe it parses well.",
        "sent_idx": 3,
        "sent_total": 4,
  }]


def test_split_sentences_simple():
  input_data = get_documents()
  expected = get_sentences()
  actual = input_data.map(
      text_util.split_sentences,
  ).flatten().compute()
  assert actual == expected


text_util.init_analyze_sentence(
    scispacy_version="en_core_sci_lg"
)

def test_analyze_sentence():
  input_data = get_sentences()
  actual = text_util.analyze_sentence(
    sent_elem=input_data[0],
    text_field="sent_text",
  )
  # Corresponds to "Document 1"
  assert "tokens" in actual
  assert len(actual["tokens"]) == 2
  assert "entities" in actual


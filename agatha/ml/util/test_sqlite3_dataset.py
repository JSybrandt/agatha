from agatha.util.sqlite3_lookup import Sqlite3LookupTable
from agatha.util.test_sqlite3_lookup import make_sqlite3_db
from agatha.ml.util import sqlite3_dataset

def test_len():
  data = {
      "A": "1",
      "B": "2",
      "C": "1",
      "D": "2",
  }
  db_path = make_sqlite3_db("test_len", data)
  table = Sqlite3LookupTable(db_path)
  dataset = sqlite3_dataset.Sqlite3Dataset(table)
  assert len(dataset) == len(data)

def test_getitem():
  expected = {
      "A": "1",
      "B": "2",
      "C": "1",
      "D": "2",
  }
  db_path = make_sqlite3_db("test_getitem", expected)
  table = Sqlite3LookupTable(db_path)
  dataset = sqlite3_dataset.Sqlite3Dataset(table)
  actual = {}
  for idx in range(len(dataset)):
    k, v = dataset[idx]
    actual[k] = v
  assert expected == actual

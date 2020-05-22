from agatha.util.test_sqlite3_lookup import make_sqlite3_db
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
import h5py

TEST_EMB_TYPE="X"
EPS = 0.00001

RawEmbedding = List[int]
RawEmbeddingTable = Dict[str, RawEmbedding]

def assert_writable(path:Path)->None:
  assert not path.exists(), f"Refusing to overwrite: {path}"
  assert path.parent.is_dir(), f"Cannot find {path.parent}"

def make_embedding_hdf5s(
    part2embs:List[List[RawEmbedding]],
    embedding_dir:Path
)->None:
  """
  This function creates an embedding hdf5 file for test purposes.
  """
  for part_idx, embeddings in enumerate(part2embs):
    path = embedding_dir.joinpath(
        f"embeddings_{TEST_EMB_TYPE}_{part_idx}.v1.h5"
    )
    assert_writable(path)
    with h5py.File(str(path), mode="w") as emb_file:
      emb_file["embeddings"] = embeddings

def make_entity_lookup_table(part2names:List[List[str]], test_dir:Path)->Path:
  """
  Writes embedding location database
  """
  data = {}
  for part_idx, names in enumerate(part2names):
    for row_idx, name in enumerate(names):
      data[name] = {
          "type": TEST_EMB_TYPE,
          "part": part_idx,
          "row": row_idx,
      }

  return make_sqlite3_db(
      test_name="entities",
      tmp_dir=test_dir,
      data=data,
  )

def setup_embedding_lookup_data(
    name2vec:RawEmbeddingTable,
    test_name:str,
    num_parts:int,
    test_root_dir:Path=Path("/tmp")
)->Tuple[Path, Path]:
  """
  Creates an embedding hdf5 file and an entity sqlite3 database for testing

  Args:
    name2vec: name2vec[x] = embedding of x
    test_name: A unique name for this test
    num_parts: The number of partitions to split this dataset among.

  Returns:
    embedding_dir, entity_db_path
    You can run EmbeddingLookupTable(*setup_embedding_lookup_data(...))
  """
  test_dir = test_root_dir.joinpath(test_name)
  if test_dir.is_dir():
    shutil.rmtree(test_dir)
  assert not test_dir.exists(), f"Refusing to overwrite {test_dir}"
  test_dir.mkdir()
  assert test_dir.is_dir(), f"Failed to make {test_dir}"

  embedding_dir = test_dir.joinpath("embeddings")
  embedding_dir.mkdir()
  assert embedding_dir.is_dir(), f"Failed to make {embedding_dir}"

  part2name2vec = [{} for _ in range(num_parts)]
  part2names = [[] for _ in range(num_parts)]
  part2embs = [[] for _ in range(num_parts)]
  for idx, (name, vec) in enumerate(name2vec.items()):
    part_idx = idx % num_parts
    part2names[part_idx].append(name)
    part2embs[part_idx].append(vec)

  make_embedding_hdf5s(part2embs, embedding_dir)
  entity_db_path = make_entity_lookup_table(part2names, test_dir)
  return embedding_dir, entity_db_path

def assert_table_contains_embeddings(
    actual=RawEmbeddingTable,
    expected=EmbeddingLookupTable,
)->None:
  assert set(actual.keys()) == set(expected.keys())
  for key in actual.keys():
    actual_vec = actual[key]
    expected_vec = expected[key]
    assert len(actual_vec) == len(expected_vec)
    # We are iterating both vectors because they might be in different formats
    for actual_val, expected_val in zip(actual_vec, expected_vec):
      # Assert that the actual is close to the expected
      assert abs(actual_val - expected_val) < EPS


def test_setup_lookup_data():
  expected = {
      "A": [1, 2, 3],
      "B": [2, 3, 4],
      "C": [5, 6, 7],
  }
  actual = EmbeddingLookupTable(*setup_embedding_lookup_data(
    expected,
    test_name="test_setup_lookup_data",
    num_parts=1
  ))
  assert_table_contains_embeddings(actual, expected)

def test_setup_lookup_data_two_parts():
  expected = {
      "A": [1, 2, 3],
      "B": [2, 3, 4],
      "C": [5, 6, 7],
  }
  actual = EmbeddingLookupTable(*setup_embedding_lookup_data(
    expected,
    test_name="test_setup_lookup_data_two_parts",
    num_parts=2
  ))
  assert_table_contains_embeddings(actual, expected)


def test_typical_embedding_lookup():
  data = {
      "A": [1, 2, 3],
      "B": [2, 3, 4],
      "C": [5, 6, 7],
  }
  embeddings = EmbeddingLookupTable(*setup_embedding_lookup_data(
    data,
    test_name="test_typical_embedding_lookup",
    num_parts=2,
  ))
  assert "A" in embeddings
  assert list(embeddings["A"]) == data["A"]

  assert "D" not in embeddings


def test_embedding_keys():
  data = {
      "A": [1, 2, 3],
      "B": [2, 3, 4],
      "C": [5, 6, 7],
      "D": [6, 7, 8],
      "E": [7, 8, 9],
  }
  embeddings = EmbeddingLookupTable(*setup_embedding_lookup_data(
    data,
    test_name="test_embedding_keys",
    num_parts=2,
  ))
  assert set(embeddings.keys()) == set(data.keys())

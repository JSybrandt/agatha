#!/usr/bin/env python3
"""
This tool converts Edge2Vec output for usage in Agatha

Edge2Vec writes vectors in a textfile with the following format

```
<num vectors> <dim>
<idx> <vector>
<idx> <vector>
...
<idx> <vector>
```

"""

from agatha.ml.util.test_embedding_lookup import (
    setup_embedding_lookup_data
)
from pathlib import Path
from fire import Fire
import pickle
from typing import Iterable, List, Tuple, Dict
from tqdm import tqdm
from agatha.ml.hypothesis_predictor.predicate_util import clean_coded_term


def iterate_vectors(
    vector_text_path:Path,
    idx2node:Dict[str, int],
    predicate_keys:Iterable[str],
)->Iterable[Tuple[str, List[float]]]:
  assert vector_text_path.is_file()
  num_vectors = None
  expected_dim = None
  with open(vector_text_path, 'r') as vec_file:
    for line in vec_file:
      tokens = line.strip().split()
      if len(tokens) == 2:
        num_vectors, expected_dim = [int(t) for t in tokens]
      if expected_dim is not None and len(tokens) == expected_dim + 1:
        idx = int(tokens[0])
        vec = [float(t) for t in tokens[1:]]
        yield clean_coded_term(idx2node[idx]), vec
  assert expected_dim is not None
  for pred_key in predicate_keys:
    yield pred_key, [0]*expected_dim


def main(
    input_vector_text_path:Path,
    input_index_path:Path,
    output_dir:Path,
)->None:
  input_vector_text_path = Path(input_vector_text_path)
  assert input_vector_text_path.is_file()

  input_index_path = Path(input_index_path)
  assert input_index_path.is_file()

  # Need an empty output dir
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  assert len(list(output_dir.iterdir())) == 0

  with open(input_index_path, 'rb') as pkl_file:
    index = pickle.load(pkl_file)

  idx2node = {i: n for n, i in index["node2idx"].items()}

  node2vec = dict(tqdm(iterate_vectors(
    input_vector_text_path,
    idx2node=idx2node,
    predicate_keys=index["predicate_keys"]
  )))

  setup_embedding_lookup_data(
      node2vec,
      test_name="edge2vec",
      num_parts=1,
      test_root_dir=output_dir,
  )


if __name__ == "__main__":
  Fire(main)

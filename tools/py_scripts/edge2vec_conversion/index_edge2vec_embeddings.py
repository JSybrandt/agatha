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
from typing import Iterable, List


def iterate_vectors(vector_text_path:Path)->Iterable[str, List[float]]:
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
        vec = [float(t) for t in tokens]
        yield idx, vec


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
  assert len(list(output_dir.listdir())) == 0

  with open(input_index_path, 'rb') as pkl_file:
    node2idx = pickle.load(pkl_file)["node2idx"]

  idx2node = {i: n for n, i in node2idx.items()}

  node2vec = {
      idx2node[idx]: vec
      for idx, vec
      in iterate_vectors(input_vector_text_path)
  }
  setup_embedding_lookup_data(
      node2vec,
      test_name="edge2vec",
      num_parts=1,
      test_root_dir=output_dir,
  )


if __name__ == "__main__":
  Fire(main)

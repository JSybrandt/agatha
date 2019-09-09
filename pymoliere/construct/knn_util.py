# https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-distribute-index-building-on-several-machines
import pandas as pd
import dask.dataframe as ddf
import faiss
from pathlib import Path
import numpy as np

def get_knn_index(
    sent_vec_df:ddf.DataFrame,
    num_neighbors:int,
    num_cells:int,
    num_probes:int,
    training_sample_rate:float,
    shared_scratch:Path,
)->faiss.Index:
  """
  Given a dataframe of idx:(NAME, VEC) return a FAISS index.
  """
  # First off, we need to get a sample to train on...

  print("Loading training sample...")
  sent_vec_df = sent_vec_df.persist()
  training_data = sent_vec_df.sample(
      frac=training_sample_rate
  ).compute()
  training_data = np.array(training_data.embedding.values.tolist())
  initial_index = faiss.IndexIVFScalarQuantizer(
      faiss.IndexFlatIP(dim),
      dim,
      num_cells,
  )
  initial_index.train(training_data)
  return initial_index

from pymoliere.ml.point_cloud_evaluator import dataset
import string
import dask
from sqlitedict import SqliteDict
import dask.bag as dbag
from distributed import Client
from argparse import ArgumentParser, Namespace
from pymoliere.ml.util.embedding_index import EmbeddingIndex
from pymoliere.util.entity_index import EntityIndex
from pymoliere.util.sqlite3_graph import Sqlite3Graph
from pymoliere.ml.point_cloud_evaluator.point_cloud_evaluator import dataset
from pymoliere.construct import dask_process_global as dpg
from pymoliere.util import database_util as dbu
from typing import List, Iterable
from pathlib import Path
import random


def configure_argument_parser(parser:ArgumentParser)->ArgumentParser:
  parser.add_argument("--cluster-address")
  parser.add_argument("--npartitions", type=int, default=100)
  return parser


def sentence_indices_to_point_cloud_db(
    start_idx:int,
    end_idx:int,
    args:Namespace,
)->Iterable[Path]:
  print("Launching sentence_indices_to_point_cloud_db")
  graph_index = Sqlite3Graph(args.sqlite_graph)
  embedding_index = EmbeddingIndex(
      embedding_dir=args.embedding_dir,
      emb_loc_db_path=args.sqlite_embedding_location,
  )
  #This one is NOT a context manager
  point_cloud_dataset = dataset.PointCloudDataset(
      embedding_dim=args.dim,
      entity_dir=args.entity_dir,
      embedding_index=embedding_index,
      graph_index=graph_index,
      source_node_type=dbu.SENTENCE_TYPE,
      neighbor_cloud_type=dbu.LEMMA_TYPE,
  )
  r_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
  db_path = args.training_data_dir.joinpath(r_name + ".sqlitedict")
  sqldict = SqliteDict(db_path, journal_mode="OFF", flag="n")
  with graph_index, embedding_index, sqldict:
    for local_idx, global_idx in enumerate(range(start_idx, end_idx)):
      sqldict[str(local_idx)] = point_cloud_dataset[local_idx]
      if local_idx % 1000 == 999:
        print("Intermediate commit:", local_idx)
        sqldict.commit()
    sqldict.commit()


def prep_training_data(args:Namespace)->None:
  if args.cluster_address is not None:
    client = Client(f"{args.cluster_address}:8786")
  else:
    client = None

  sent_idx = EntityIndex(args.entity_dir, dbu.SENTENCE_TYPE)
  num_sentences = len(sent_idx)
  sent_per_task = int(num_sentences / args.npartitions)
  print(f"Sentences in total: {num_sentences}. Per task: {sent_per_task}")
  tasks = [
      [sent_per_task*i, sent_per_task*i+1] for i in range(args.npartitions)
  ]
  tasks[-1][-1] = num_sentences
  tasks = []
  for task_idx in range(args.npartitions):
    start_idx = sent_per_task*task_idx
    end_idx = start_idx + sent_per_task
    tasks.append(
        dask.delayed(sentence_indices_to_point_cloud_db)(
          start_idx, end_idx, args
        )
    )
  print("Launching dask dask to process training data")
  dask.compute(tasks)

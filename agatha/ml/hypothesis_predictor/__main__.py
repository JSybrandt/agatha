from argparse import ArgumentParser, Namespace
from agatha.ml.hypothesis_predictor import hypothesis_predictor as hp
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path


def train(
    training_args:Namespace,
    graph_db:Path,
    entity_db:Path,
    embedding_dir:Path,
):
  assert graph_db.is_file()
  assert entity_db.is_file()
  assert embedding_dir.is_dir()
  trainer = Trainer.from_argparse_args(training_args)
  model = hp.HypothesisPredictor(training_args)
  model.prepare_for_training()
  trainer.fit(model)

if __name__ == "__main__":
  parser = ArgumentParser()
  parser = Trainer.add_argparse_args(parser)
  parser = hp.HypothesisPredictor.add_argparse_args(parser)
  # These arguments will be serialized with the model
  training_args = parser.parse_known_args()

  # These arguments will be forgotten after training is complete
  parser.add_argument(
      "--graph-db",
      type=Path,
      help="Location of graph sqlite3 lookup table."
  )
  parser.add_argument(
      "--entity-db",
      type=Path,
      help="Location of entity sqlite3 lookup table."
  )
  parser.add_argument(
      "--embedding-dir",
      type=Path,
      help="Location of graph embedding hdf5 files."
  )
  all_args = parser.parse_args()

  train(
      training_args=training_args,
      graph_db=all_args.graph_db,
      entity_db=all_args.entity_db,
      embedding_dir=all_args.embedding_dir
  )

import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import torch
from pymoliere.ml.hypothesis_predictor.dataset import (
    VERB2IDX,
    PredicateLoader,
    predicate_collate,
    TestPredicateLoader,
    HypothesisBatch,
)
from typing import Dict, Any
from pathlib import Path
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb
from pymoliere.ml.util.embedding_index import EmbeddingIndex
from pymoliere.ml.util.entity_index import EntityIndex
import pymoliere.util.database_util as dbu
from copy import deepcopy
from pymoliere.util.sqlite3_graph import Sqlite3Graph

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams
    # Helper data structures
    self.embedding_index = EmbeddingIndex(
        embedding_dir=self.hparams.embedding_dir,
        emb_loc_db_path=self.hparams.sqlite_embedding_path,
    ).__enter__()
    self.graph_index = Sqlite3Graph(self.hparams.sqlite_graph_path).__enter__()

    # All predicates, will split
    predicates = PredicateLoader(
      embedding_index=self.embedding_index,
      graph_index=self.graph_index,
      entity_dir=self.hparams.entity_dir,
      neighbors_per_term=self.hparams.neighbors_per_term,
    )
    # Validation samples
    val_size = int(len(predicates)*self.hparams.validation_fraction)
    train_size = len(predicates) - val_size
    # split the dataset in two uneven parts
    self.training_data, self.val_data = torch.utils.data.random_split(
        predicates, [train_size, val_size]
    )
    # configure test
    self.test_data = TestPredicateLoader(
        self.hparams.test_data_dir,
        self.embedding_index,
        self.graph_index,
    )
    # Layers
    # This is going to transform into {-1, 1}
    self.embedding_transformation = torch.nn.Linear(
        self.hparams.dim, self.hparams.dim
    )
    # Extra
    self.loss_fn = torch.nn.BCELoss()

  def _batch_to_device(self, b:HypothesisBatch)->HypothesisBatch:
    device = next(self.parameters()).device
    return HypothesisBatch(
        subject_embedding=b.subject_embedding.to(device),
        object_embedding=b.object_embedding.to(device),
        subject_neighbor_embeddings=b.subject_neighbor_embeddings.to(device),
        object_neighbor_embeddings=b.object_neighbor_embeddings.to(device),
        label=b.label.to(device),
    )

  def forward(self, batch:HypothesisBatch)->torch.FloatTensor:
    # batch X dim
    local_subj_emb = self.embedding_transformation(batch.subject_embedding)
    local_subj_emb = torch.tanh(local_subj_emb)
    # batch X dim
    local_obj_emb = self.embedding_transformation(batch.object_embedding)
    local_obj_emb = torch.tanh(local_obj_emb)
    # # neigh_seq X batch X dim
    # local_subj_neig_embs = self.embedding_transformation(
        # batch.subject_neighbor_embeddings
    # )
    # local_subj_neig_embs = torch.tanh(local_subj_neig_embs)
    # # neigh_seq X batch X dim
    # local_obj_neig_embs = self.embedding_transformation(
        # batch.object_neighbor_embeddings
    # )
    # local_obj_neig_embs = torch.tanh(local_obj_neig_embs)
    # |batch|
    subj_obj_dots = (local_subj_emb * local_obj_emb).sum(dim=1)
    return torch.sigmoid(subj_obj_dots)


  def training_step(self, batch, batch_idx):
    batch = self._batch_to_device(batch)
    predictions = self.forward(batch)
    device = next(self.parameters()).device
    metrics=dict(
        loss=self.loss_fn(predictions, batch.label),
        accuracy=(
          ((predictions > 0.5) == (batch.label > 0.5))
          .sum().float() / float(len(predictions))
        ),
    )
    return {
        'loss': metrics["loss"],
        'progress_bar': metrics,
        'log': metrics,
    }

  def validation_step(self, batch, batch_idx):
    return self.training_step(batch, batch_idx)["log"]

  def test_step(self, batch, batch_idx):
    return self.training_step(batch, batch_idx)["log"]

  def _on_end(self, outputs):
    metrics = {}
    for metric in outputs[0]:
      metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()
    return metrics

  def validation_end(self, outputs):
    return self._on_end(outputs)

  def test_end(self, outputs):
    return self._on_end(outputs)

  def _config_dl(self, dataset):
    shuffle=True
    sampler=None
    if self.hparams.distributed:
      shuffle=False
      sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    collate = lambda batch: predicate_collate(
        positive_samples=batch,
        num_negative_samples=int(
          len(batch) * self.hparams.negative_sample_ratio
        ),
        neighbors_per_term=self.hparams.neighbors_per_term,
    )
    return torch.utils.data.DataLoader(
        dataset=self.training_data,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=self.hparams.batch_size,
        collate_fn=collate,
    )

  @pl.data_loader
  def train_dataloader(self):
    return self._config_dl(self.training_data)

  @pl.data_loader
  def val_dataloader(self):
    return self._config_dl(self.val_data)

  @pl.data_loader
  def test_dataloader(self):
    return self._config_dl(self.test_data)

  def configure_optimizers(self):
    return Lamb(
        self.parameters(),
        # facebook paper says linear growth with batch size
        lr=self.hparams.lr,
        weight_decay=0.01,
    )

  def optimizer_step(
      self,
      epoch_idx,
      batch_idx,
      optimizer,
      optimizer_idx,
      second_order_closure=None
  ):
    # warm up lr
    if  self.trainer.global_step < self.hparams.warmup_steps:
      lr_scale = min(
          1.,
          float(self.trainer.global_step + 1)/float(self.hparams.warmup_steps)
      )
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale * self.hparams.lr
    optimizer.step()
    optimizer.zero_grad()

  @staticmethod
  def configure_argument_parser(parser:ArgumentParser)->ArgumentParser:
    parser.add_argument(
        "--sqlite-embedding-path",
        help="Location of the db containing references for node's embeddings."
    )
    parser.add_argument(
        "--sqlite-graph-path",
        help="Location of the db containing references for node's embeddings."
    )
    parser.add_argument(
        "--embedding-dir",
        help="Location of the directory containing H5 files, following PTBG"
    )
    parser.add_argument(
        "--entity-dir",
        help="Location of the directory containing json and count files."
    )
    parser.add_argument(
        "--test-data-dir",
        help="Directory containing published.txt and noise.txt"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--negative-sample-ratio",
        type=float,
        default=5,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--neighbors-per-term",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
    )
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train-num-machines", type=int, default=1)
    return parser

  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size*self.hparams.train_num_machines
    )

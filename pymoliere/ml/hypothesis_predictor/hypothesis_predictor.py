import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import torch
from pymoliere.ml.hypothesis_predictor.dataset import (
    VERB2IDX,
    PredicateLoader,
    predicate_collate,
)
from typing import Dict, Any
from pathlib import Path

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams
    self.subj_dense = torch.nn.Linear(self.hparams.dim, self.hparams.dim)
    self.obj_dense = torch.nn.Linear(self.hparams.dim, self.hparams.dim)
    self.joint_dense = torch.nn.Linear(self.hparams.dim*2, self.hparams.dim)
    self.joint_to_logit = torch.nn.Linear(self.hparams.dim, 1)
    self.loss_fn = torch.nn.BCELoss()
    self.predicate_dataset = PredicateLoader(
      entity_dir=self.hparams.entity_dir,
      embedding_dir=self.hparams.embedding_dir,
    )

  def forward(self, model_in:Dict[str, Any])->torch.FloatTensor:
    device = next(self.parameters()).device
    subjects = model_in["subjects"].to(device)
    objects = model_in["objects"].to(device)
    # batch X dim
    subjects = torch.relu(self.subj_dense(subjects))
    # batch X dim
    objects = torch.relu(self.obj_dense(objects))
    # batch X dim  Goes from 2Xdim to dim
    joint = torch.relu(self.joint_dense(torch.cat((subjects, objects), dim=1)))
    # batch X 1
    return torch.sigmoid(self.joint_to_logit(joint)).reshape(-1)

  def training_step(self, batch, batch_idx):
    device = next(self.parameters()).device
    predictions = self.forward(batch)
    labels = batch["labels"].to(device)
    loss = self.loss_fn(predictions, labels)
    metrics=dict(
        loss=loss,
        accuracy=((predictions>0.5)==(labels > 0.5)).sum()/float(len(predictions)),
    )
    return {
        'loss': loss,
        'progress_bar': metrics,
        'log': metrics,
    }

  @pl.data_loader
  def train_dataloader(self):
    if self.hparams.debug:
      sampler = None
    else:
      sampler=torch.utils.data.distributed.DistributedSampler(self.predicate_dataset)
    return torch.utils.data.DataLoader(
        dataset=self.predicate_dataset,
        sampler=sampler,
        batch_size=int(self.hparams.batch_size/2),
        # Collate will double the batch
        collate_fn=predicate_collate,
    )

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

  @staticmethod
  def configure_argument_parser(parser:ArgumentParser)->ArgumentParser:
    parser.add_argument(
        "--sqlite-embedding-location",
        type=Path,
        help="Location of the db containing references for node's embeddings."
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        help="Location of the directory containing H5 files, following PTBG"
    )
    parser.add_argument(
        "--entity-dir",
        type=Path,
        help="Location of the directory containing json and count files."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
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
        "--model-mode",
        type=str,
        default="simple",
        choices=["simple", "with_verb"],
    )
    parser.add_argument("--debug", action="store_true")
    return parser

  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size
    )

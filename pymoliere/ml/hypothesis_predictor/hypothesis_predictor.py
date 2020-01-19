import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import torch
from pymoliere.ml.hypothesis_predictor.dataset import (
    VERB2IDX,
    PredicateLoader,
    train_predicate_collate,
    test_predicate_collate,
    TestPredicateLoader, 
)
from typing import Dict, Any
from pathlib import Path
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb
from pymoliere.ml.util.embedding_index import (
    EmbeddingIndex
)

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams
    self.h1 = torch.nn.Linear(self.hparams.dim*2, self.hparams.dim*2)
    self.h2 = torch.nn.Linear(self.hparams.dim*2, self.hparams.dim)
    self.h3 = torch.nn.Linear(self.hparams.dim, int(self.hparams.dim/2))
    self.predict_label = torch.nn.Linear(int(self.hparams.dim/2), 1)
    self.predict_verb = torch.nn.Linear(int(self.hparams.dim/2), len(VERB2IDX))

    self.label_loss_fn = torch.nn.BCELoss()
    self.verb_loss_fn = torch.nn.NLLLoss()
    # TRAIN MODE
    if self.hparams.mode == "train":
      predicate_dataset = PredicateLoader(
        entity_dir=self.hparams.entity_dir,
        embedding_dir=self.hparams.embedding_dir,
      )
      val_size = int(len(predicate_dataset)*self.hparams.validation_fraction)
      train_size = len(predicate_dataset) - val_size
      self.train_predicates, self.val_predicates = \
          torch.utils.data.random_split(
            predicate_dataset, [train_size, val_size]
          )
    else:
      self.train_predicates = []
      self.val_predicates = []
    # TEST MODE
    if self.hparams.mode == "test":
      with EmbeddingIndex(
            embedding_dir=self.hparams.embedding_dir,
            emb_loc_db_path=self.hparams.sqlite_embedding_location,
      ) as embedding_index:
        self.test_predicates = TestPredicateLoader(
          self.hparams.test_data_dir,
          embedding_index,
        )
    else:
      self.test_predicates = []

  def forward(self, model_in:Dict[str, Any])->torch.FloatTensor:
    device = next(self.parameters()).device
    subjects = model_in["subjects"].to(device)
    objects = model_in["objects"].to(device)
    joint = torch.cat((subjects, objects), dim=1)
    joint = torch.relu(self.h1(joint))
    joint = torch.relu(self.h2(joint))
    joint = torch.relu(self.h3(joint))

    predicted_labels = self.predict_label(joint)
    predicted_labels = torch.sigmoid(predicted_labels).reshape(-1)
    predicted_verbs = self.predict_verb(joint)
    predicted_verbs = torch.log_softmax(predicted_verbs, dim=1)
    return dict(
        labels=predicted_labels.reshape(-1),
        verbs=predicted_verbs,
    )

  def training_step(self, batch, batch_idx):
    predictions = self.forward(batch)
    device = next(self.parameters()).device
    true_labels = batch["labels"].to(device)
    true_verbs = batch["verbs"].to(device)
    label_loss = self.label_loss_fn(predictions["labels"], true_labels)
    verb_loss = self.verb_loss_fn(predictions["verbs"], true_verbs)
    loss = label_loss + verb_loss
    label_accuracy = (
        ((predictions["labels"] > 0.5) == (true_labels > 0.5)).sum()
        / float(len(predictions["labels"]))
    )
    verb_accuracy = (
        (predictions["verbs"].argmax(dim=1) == true_verbs).sum()
        / float(len(predictions["verbs"]))
    )
    metrics=dict(
        loss=loss,
        label_loss=label_loss,
        verb_loss=verb_loss,
        label_accuracy=label_accuracy,
        verb_accuracy=verb_accuracy,
    )
    return {
        'loss': loss,
        'progress_bar': metrics,
        'log': metrics,
    }

  def validation_step(self, batch, batch_idx):
    predictions = self.forward(batch)
    device = next(self.parameters()).device
    true_labels = batch["labels"].to(device)
    true_verbs = batch["verbs"].to(device)
    label_loss = self.label_loss_fn(predictions["labels"], true_labels)
    verb_loss = self.verb_loss_fn(predictions["verbs"], true_verbs)
    loss = label_loss + verb_loss
    label_accuracy = (
        ((predictions["labels"] > 0.5) == (true_labels > 0.5)).sum()
        / float(len(predictions["labels"]))
    )
    verb_accuracy = (
        (predictions["verbs"].argmax(dim=1) == true_verbs).sum()
        / float(len(predictions["verbs"]))
    )
    return dict(
        loss=loss,
        label_loss=label_loss,
        verb_loss=verb_loss,
        label_accuracy=label_accuracy,
        verb_accuracy=verb_accuracy,
    )

  def validation_end(self, outputs):
    mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
    mean_label_accruacy = torch.stack([x["label_accuracy"] for x in outputs]).mean()
    mean_verb_accruacy = torch.stack([x["verb_accuracy"] for x in outputs]).mean()
    vals = dict(
        val_loss=mean_loss,
        label_accruacy=mean_label_accruacy,
        verb_accruacy=mean_verb_accruacy,
    )
    print(vals)
    return vals


  def test_step(self, batch, batch_idx):
    predictions = self.forward(batch)
    device = next(self.parameters()).device
    true_labels = batch["labels"].to(device)
    true_verbs = batch["verbs"].to(device)
    label_loss = self.label_loss_fn(predictions["labels"], true_labels)
    verb_loss = self.verb_loss_fn(predictions["verbs"], true_verbs)
    loss = label_loss + verb_loss
    label_accuracy = (
        ((predictions["labels"] > 0.5) == (true_labels > 0.5)).sum()
        / float(len(predictions["labels"]))
    )
    verb_accuracy = (
        (predictions["verbs"].argmax(dim=1) == true_verbs).sum()
        / float(len(predictions["verbs"]))
    )
    return dict(
        loss=loss,
        label_loss=label_loss,
        verb_loss=verb_loss,
        label_accuracy=label_accuracy,
        verb_accuracy=verb_accuracy,
    )

  def test_end(self, outputs):
    mean_loss = torch.stack([x["loss"] for x in outputs]).mean()
    mean_label_accruacy = torch.stack([x["label_accuracy"] for x in outputs]).mean()
    mean_verb_accruacy = torch.stack([x["verb_accuracy"] for x in outputs]).mean()
    vals = dict(
        test_loss=mean_loss,
        label_accruacy=mean_label_accruacy,
        verb_accruacy=mean_verb_accruacy,
    )
    print(vals)
    return vals

  @pl.data_loader
  def train_dataloader(self):
    if self.hparams.distributed:
      shuffle=False
      sampler=torch.utils.data.distributed.DistributedSampler(self.train_predicates)
    else:
      sampler = None
      shuffle=True
    return torch.utils.data.DataLoader(
        dataset=self.train_predicates,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=int(self.hparams.batch_size/2),
        # Collate will double the batch
        collate_fn=train_predicate_collate,
        pin_memory=True,
    )

  @pl.data_loader
  def val_dataloader(self):
    if self.hparams.distributed:
      sampler=torch.utils.data.distributed.DistributedSampler(self.val_predicates)
    else:
      sampler = None
    return torch.utils.data.DataLoader(
        dataset=self.val_predicates,
        sampler=sampler,
        batch_size=int(self.hparams.batch_size/2),
        # Collate will double the batch
        collate_fn=train_predicate_collate,
        pin_memory=True,
    )

  @pl.data_loader
  def test_dataloader(self):
    if self.hparams.distributed:
      sampler=torch.utils.data.distributed.DistributedSampler(self.test_predicates)
    else:
      sampler = None
    return torch.utils.data.DataLoader(
        dataset=self.test_predicates,
        sampler=sampler,
        batch_size=self.hparams.batch_size,
        # Collate will double the batch
        collate_fn=test_predicate_collate,
        pin_memory=True,
    )

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
        "--warmup-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--test-data-dir",
        type=Path,
        help="Directory containing published.txt and noise.txt"
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
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

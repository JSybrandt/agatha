from collections import defaultdict
import torch
import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
from agatha.ml.util.lamb_optimizer import Lamb
from typing import List, Tuple, Dict, Any
from agatha.util.entity_types import UMLS_TERM_TYPE, PREDICATE_TYPE
from agatha.util.sqlite3_lookup import Sqlite3Graph
from agatha.ml.util.embedding_lookup import EmbeddingLookupTable
from agatha.util.misc_util import iter_to_batches
from agatha.ml.hypothesis_predictor import predicate_util
from pathlib import Path
import os

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams

    self.verbose = False
    self.distributed = False

    # Layers
    ## Graph Emb Input
    self.embedding_transformation = torch.nn.Linear(
        self.hparams.dim, self.hparams.dim
    )
    ## Encoder Stack
    self.encode_predicate_data = torch.nn.TransformerEncoder(
        encoder_layer=torch.nn.TransformerEncoderLayer(
          self.hparams.dim,
          self.hparams.transformer_heads,
          self.hparams.transformer_ff_dim,
          self.hparams.transformer_dropout,
          "relu"
        ),
        num_layers=self.hparams.transformer_layers,
        norm=torch.nn.LayerNorm(self.hparams.dim),
    )
    ## Avg emb to logit
    self.encoding_to_logit = torch.nn.Linear(
      self.hparams.dim, 1
    )
    # Loss Fn
    self.loss_fn = torch.nn.MarginRankingLoss(margin=self.hparams.margin)

    # Helper data, set by configure_paths
    self.embeddings = None
    self.graph = None
    # Helper data, set by prepare_for_training
    self.training_predicates = None
    self.validation_predicates = None
    self.predicates = None
    self.coded_terms = None
    self.predicate_batch_generator = None

  def _vprint(self, *args, **kwargs):
    if self.verbose:
      print(*args, **kwargs)

  def configure_paths(
      self,
      graph_db:Path,
      entity_db:Path,
      embedding_dir:Path,
  ):
    assert graph_db.is_file(), f"Failed to find {graph_db}"
    assert entity_db.is_file(), f"Failed to find {entity_db}"
    assert embedding_dir.is_dir(), f"Failed to find {embedding_dir}"
    self.embeddings = EmbeddingLookupTable(
        embedding_dir=embedding_dir,
        entity_db=entity_db,
    )
    self.graph=Sqlite3Graph(graph_db)

  def paths_set(self)->bool:
    return self.embeddings is not None and self.graph is not None

  def preload(self, include_embeddings:bool=False)->None:
    assert self.paths_set(), "Must call configure_paths before preload."
    if not self.is_preloaded():
      self.graph.preload()
      if include_embeddings:
        self.embeddings.preload()
      else:
        self.embeddings.entities.preload()

  def is_preloaded(self)->None:
    return (
        self.paths_set()
        and self.graph.is_preloaded()
        and self.embeddings.entities.is_preloaded()
    )

  def prepare_for_training(self)->None:
    assert self.paths_set(), \
        "Must call configure_paths before prepare_for_training"
    entities = self.embeddings.keys()
    assert len(entities) > 0, "Failed to find embedding entities."
    self.coded_terms = list(filter(
      lambda k: k[0] == UMLS_TERM_TYPE,
      entities
    ))
    self.predicates = list(filter(
      lambda k: k[0] == PREDICATE_TYPE,
      entities
    ))
    self._vprint("Splitting train/validation")
    validation_size = int(
        len(self.predicates) * self.hparams.validation_fraction
    )
    training_size = len(self.predicates) - validation_size
    self._vprint("\t- Training:", training_size)
    self._vprint("\t- Validation:", validation_size)
    (
        self.training_predicates,
        self.validation_predicates,
    ) = torch.utils.data.random_split(
        self.predicates,
        [training_size, validation_size]
    )
    self._vprint("Preparing Batch Generator")
    self.predicate_batch_generator = predicate_util.PredicateBatchGenerator(
        graph=self.graph,
        embeddings=self.embeddings,
        predicates=self.predicates,
        coded_terms=self.coded_terms,
        neighbor_sample_rate=self.hparams.neighbor_sample_rate,
        negative_swap_rate=self.hparams.negative_swap_rate,
        negative_scramble_rate=self.hparams.negative_scramble_rate,
    )
    self._vprint("Ready for training!")

  def _configure_dataloader(
      self,
      predicate_dataset:torch.utils.data.Dataset,
      shuffle:bool,
  )->torch.utils.data.DataLoader:
    self.preload()
    sampler = None
    if self.distributed:
      shuffle = False
      sampler=torch.utils.data.distributed.DistributedSampler(predicate_dataset)
    return torch.utils.data.DataLoader(
        dataset=predicate_dataset,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=self.hparams.positives_per_batch,
        num_workers=self.hparams.dataloader_workers,
        pin_memory=True,
    )

  def train_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Training Dataloader")
    return self._configure_dataloader(
        self.training_predicates,
        shuffle=True,
    )

  def val_dataloader(self)->torch.utils.data.DataLoader:
    self._vprint("Getting Validation Dataloader")
    return self._configure_dataloader(
        self.validation_predicates,
        shuffle=False,
    )

  def forward(self, predicate_embeddings:torch.FloatTensor)->torch.FloatTensor:
    local_stacked_emb = self.embedding_transformation(predicate_embeddings)
    local_stacked_emb = torch.relu(local_stacked_emb)
    encoded_predicate = self.encode_predicate_data(local_stacked_emb)
    encoded_predicate = encoded_predicate.mean(dim=0)
    logit = self.encoding_to_logit(encoded_predicate)
    logit = torch.sigmoid(logit)
    return logit.reshape(-1)

  def get_device(self):
    return next(self.parameters()).device

  def _step(self, positive_predicates:List[str])->Dict[str, Any]:
    pos, negs = self.predicate_batch_generator.generate(positive_predicates)
    positive_predictions = self.forward(
        predicate_util
        .collate_predicate_embeddings(pos)
        .to(self.get_device())
    )
    partial_losses = []
    for neg in negs:
      negative_predictions = self.forward(
          predicate_util
          .collate_predicate_embeddings(neg)
          .to(self.get_device())
      )
      partial_losses.append(
          self.loss_fn(
            positive_predictions,
            negative_predictions,
            positive_predictions.new_ones(len(positive_predictions))
          )
      )
    return dict(
        loss=sum(partial_losses)
    )

  def training_step(
      self,
      positive_predictions:List[str],
      batch_idx:int
  )->Dict[str, Any]:
    return self._step(positive_predictions)

  def validation_step(
      self,
      positive_predictions:List[str],
      batch_idx:int
  )->Dict[str, Any]:
    return self._step(positive_predictions)

  def _on_epoch_end(
      self,
      outputs:List[Dict[str,torch.Tensor]]
  )->Dict[str, Dict[str,torch.Tensor]]:
    metric2values = defaultdict(list)
    for output in outputs:
      for k, v in output.items():
        metric2values[k].append(v)
    metric2value = {}
    for k in metric2values:
      if len(metric2values[k]) > 0:
        metric2values[k] = sum(metric2values[k]) / len(metric2values[k])
    return dict(
      log=metric2values,
      progress_bar=metric2values
    )

  def validation_epoch_end(self, outputs:List[Dict[str,torch.Tensor]]):
    return self._on_epoch_end(outputs)

  def configure_optimizers(self):
    self._vprint("Configuring optimizers")
    return Lamb(
        self.parameters(),
        lr=self.hparams.lr,
        weight_decay=self.hparams.weight_decay,
    )

  def on_train_start(self):
    pl.LightningModule.on_train_start(self)
    self._vprint("Training started")


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
  def add_argparse_args(parser:ArgumentParser)->ArgumentParser:
    """
    These arguments will be serialized along with the model after training.
    Path-specific arguments will be passed in separately.
    """
    parser.add_argument("--dataloader-workers", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--negative-scramble-rate", type=int)
    parser.add_argument("--negative-swap-rate", type=int)
    parser.add_argument("--neighbor-sample-rate", type=int)
    parser.add_argument("--positives-per-batch", type=int)
    parser.add_argument("--transformer-dropout", type=float)
    parser.add_argument("--transformer-ff-dim", type=int)
    parser.add_argument("--transformer-heads", type=int)
    parser.add_argument("--transformer-layers", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    return parser

  def init_ddp_connection(
      self,
      proc_rank:int,
      world_size:int,
      is_slurm_managing_tasks:bool=False
  )->None:
    """
    Override to define your custom way of setting up a distributed environment.
    Lightning's implementation uses env:// init by default and sets the first node as root
    for SLURM managed cluster.
    Args:
        proc_rank: The current process rank within the node.
        world_size: Number of GPUs being use across all nodes. (num_nodes * num_gpus).
        is_slurm_managing_tasks: is cluster managed by SLURM.
    """
    if is_slurm_managing_tasks:
      self._init_slurm_connection()

    if 'MASTER_ADDR' not in os.environ:
      log.warning("MASTER_ADDR environment variable is not defined. Set as localhost")
      os.environ['MASTER_ADDR'] = '127.0.0.1'

    if 'MASTER_PORT' not in os.environ:
      log.warning("MASTER_PORT environment variable is not defined. Set as 12910")
      os.environ['MASTER_PORT'] = '12910'

    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != world_size:
      log.warning("WORLD_SIZE environment variable is not equal to the computed "
                   "world size. Ignored.")

    # Reverting to GLOO until nccl upgrades in pytorch are complete
    #torch_backend = "nccl" if self.trainer.on_gpu else "gloo"
    torch_backend = "gloo"

    self._vprint(
        "Attempting connection:",
        torch_backend,
        os.environ["MASTER_ADDR"], proc_rank
    )
    torch.distributed.init_process_group(
        torch_backend,
        rank=proc_rank,
        world_size=world_size
    )

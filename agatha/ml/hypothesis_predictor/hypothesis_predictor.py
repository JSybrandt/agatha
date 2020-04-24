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

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams

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

    self.hparams.batch_size = self.hparams.positives_per_batch * (
        self.hparams.neg_swap_rate + self.hparams.neg_scramble_rate
    )

    # Helper data, set by configure_paths
    self.embeddings = None
    self.graph = None
    # Helper data, set by prepare_for_training
    self.training_predicates = None
    self.validation_predicates = None
    self.predicates = None
    self.coded_terms = None
    self.predicate_batch_generator = None

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
    self.graph=Sqlite3Graph(self.graph_db)

  def is_ready(self)->bool:
    return self.embeddings is not None and self.graph is not None

  def preload(self)->None:
    assert is_ready(), "Must set graph/embeddings before preload."
    self.graph.preload()
    self.embeddings.preload()

  def is_preloaded(self)->None:
    return (
        self.is_ready()
        and self.graph.is_preloaded()
        and self.embeddings.is_preloaded()
    )

  def prepare_for_training(self)->None:
    assert self.is_ready(), "Must call configure_paths first"
    entities = self.embeddings.keys()
    self.coded_terms = list(filter(
      lambda k: k[0] == UMLS_TERM_TYPE,
      entities
    ))
    self.predicates = list(filter(
      lambda k: k[0] == PREDICATE_TYPE,
      entities
    ))
    validation_size = int(
        len(self.predicates) * self.hparams.validation_fraction
    )
    training_size = len(self.predicates) - validation_size
    (
        self.training_predicates,
        self.validation_predicates,
    ) = torch.utils.data.random_split(
        self.predicates,
        [training_size, validation_size]
    )
    self.predicate_batch_generator = predicate_util.PredicateBatchGenerator(
        graph=self.graph,
        embeddings=self.embeddings,
        predicates=self.training_predicates,
        coded_terms=self.coded_terms,
        neighbor_sample_rate=self.hparams.neighbor_sample_rate,
        negative_swap_rate=self.hparams.negative_swap_rate,
        negative_scramble_rate=self.hparams.negative_scramble_rate,
    )

  def _configure_dataloader(
      self,
      predicate_dataset:torch.utils.data.Dataset
  )->torch.utils.data.DataLoader:
    shuffle = True
    sampler = None
    if self.hparams.distributed:
      shuffle = False
      sampler=torch.utils.data.distributed.DistributedSampler(predicate_dataset)
    return torch.utils.data.DataLoader(
        dataset=predicate_dataset,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=self.hparams.positives_per_batch,
    )

  def train_dataloader(self)->torch.utils.data.DataLoader:
    return self._configure_dataloader(self.training_predicates)

  def val_dataloader(self)->torch.utils.data.DataLoader:
    return self._configure_dataloader(self.validation_predicates)

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

  def configure_optimizers(self):
    return Lamb(
        self.parameters(),
        lr=self.hparams.lr,
        weight_decay=self.hparams.weight_decay,
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


  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size*self.hparams.train_num_machines
    )

  @staticmethod
  def add_argparse_args(parser:ArgumentParser)->ArgumentParser:
    """
    These arguments will be serialized along with the model after training.
    Path-specific arguments will be passed in separately.
    """
    parser.add_argument(
        "--positives-per-batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--neg-scramble-rate",
        type=int,
        default=10,
        help="A negative scramble draws the neighborhood sets randomly"
    )
    parser.add_argument(
        "--neg-swap-rate",
        type=int,
        default=10,
        help="A negative swap exchanges the subject and object data in full."
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
        "--neighbors-per-term",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--transformer-layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer-ff-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    return parser

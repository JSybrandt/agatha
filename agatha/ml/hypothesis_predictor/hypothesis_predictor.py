import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
import torch
from agatha.ml.hypothesis_predictor.dataset import (
    VERB2IDX,
    PredicateLoader,
    predicate_collate,
    TestPredicateLoader,
    HypothesisTensors,
    observations_to_tensors,
    generate_predicate_observation,
)
from typing import List, Tuple
from agatha.util.entity_types import UMLS_TERM_TYPE, PREDICATE_TYPE
from agatha.ml.abstract_generator.lamb_optimizer import Lamb
from agatha.ml.util.embedding_index import (
    PreloadedEmbeddingIndex,
    EmbeddingIndex,
)
from agatha.util.sqlite3_graph import (
    PreloadedSqlite3Graph,
    Sqlite3Graph,
)
from agatha.util.misc_util import iter_to_batches
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

class HypothesisPredictor(pl.LightningModule):
  def __init__(self, hparams:Namespace):
    super(HypothesisPredictor, self).__init__()
    self.hparams = hparams
    assert self.hparams.neighbors_per_term > 0
    # Layers
    # This is going to transform into {-1, 1}
    self.embedding_transformation = torch.nn.Linear(
        self.hparams.dim, self.hparams.dim
    )
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
    self.encoding_to_logit = torch.nn.Linear(
      self.hparams.dim, 1
    )
    # Extra
    # backwards compatibility
    if hasattr(self.hparams, "margin"):
      margin = self.hparams.margin
    else:
      margin = 0.1
    self.loss_fn = torch.nn.MarginRankingLoss(margin=margin)
    self.hparams.batch_size = self.hparams.positives_per_batch * (
        self.hparams.neg_swap_rate + self.hparams.neg_scramble_rate
    )

  def _check_file_paths(self)->None:
    MSG = "Consider running model.set_data_root(...)"
    def assert_dir(path):
      path = Path(path)
      assert path.is_dir(), f"Failed to find directory: {path}. {MSG}"
    def assert_file(path):
      path = Path(path)
      assert path.is_file(), f"Failed to find file: {path}. {MSG}"
    assert_dir(self.hparams.embedding_dir)
    assert_file(self.hparams.sqlite_graph_path)
    assert_file(self.hparams.sqlite_embedding_path)
    self._is_forward_ready = False

  def set_data_root(self, root_dir:Path)->None:
    root_dir = Path(root_dir)
    self.hparams.embedding_dir = str(root_dir.joinpath("embeddings"))
    self.hparams.sqlite_graph_path = str(
        root_dir
        .joinpath("helper_databases")
        .joinpath("graph_predicate_subset.sqlite3")
    )
    self.hparams.sqlite_embedding_path = str(
        root_dir
        .joinpath("helper_databases")
        .joinpath("embedding_location_subset.sqlite3")
    )
    self._check_file_paths()

  def __enter__(self)->None:
    self.init()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.deinit()
    return False # Don't want to handle exceptions

  def deinit(self)->None:
    try:
      self.embedding_index.__exit__(None, None, None)
    except Exception:
      pass
    try:
      self.graph_index.__exit__(None, None, None)
    except Exception:
      pass

  def init(self)->None:
    self._check_file_paths()
    self.embedding_index = EmbeddingIndex(
        embedding_dir=self.hparams.embedding_dir,
        embedding_location_db_path=self.hparams.sqlite_embedding_path,
    ).__enter__()
    self.graph_index = Sqlite3Graph(self.hparams.sqlite_graph_path).__enter__()
    self._is_forward_ready = True

  def init_preload(self)->None:
    self._check_file_paths()
    print("Loading all graph and embedding helper data. Will take a minute.")
    # Helper data structures
    self.embedding_index = PreloadedEmbeddingIndex(
        embedding_dir=self.hparams.embedding_dir,
        embedding_location_db_path=self.hparams.sqlite_embedding_path,
        entity_types=UMLS_TERM_TYPE+PREDICATE_TYPE
    )
    self.graph_index = PreloadedSqlite3Graph(
        self.hparams.sqlite_graph_path
    ).__enter__()

    self._is_forward_ready = True

  def init_training_datasets(self)->None:
    #self.init()
    self.init_preload()
    # All predicates, will split
    predicates = PredicateLoader(
      embedding_index=self.embedding_index,
      graph_index=self.graph_index,
      neighbors_per_term=self.hparams.neighbors_per_term,
      entity_dir=self.hparams.entity_dir,
    )
    # Validation samples
    val_size = int(len(predicates)*self.hparams.validation_fraction)
    train_size = len(predicates) - val_size
    # split the dataset in two uneven parts
    self.training_data, self.val_data = torch.utils.data.random_split(
        predicates, [train_size, val_size]
    )


  def _tensors_to_device(self, b:HypothesisTensors)->HypothesisTensors:
    device = next(self.parameters()).device
    return HypothesisTensors(
        subject_embedding=b.subject_embedding.to(device),
        object_embedding=b.object_embedding.to(device),
        subject_neighbor_embeddings=b.subject_neighbor_embeddings.to(device),
        object_neighbor_embeddings=b.object_neighbor_embeddings.to(device),
        label=b.label.to(device),
    )

  def predict_from_terms(
      self, term_pairs:List[Tuple[str, str]]
  )->List[float]:
    assert self._is_forward_ready, "Must run model.init()"
    res = []
    for pair_batch in iter_to_batches(term_pairs, self.hparams.batch_size):
      model_input = observations_to_tensors([
          generate_predicate_observation(
            subj=subj,
            obj=obj,
            neighbors_per_term=self.hparams.neighbors_per_term,
            graph_index=self.graph_index,
            embedding_index=self.embedding_index,
          )
          for subj, obj in term_pairs
      ])
      res += list(self.forward(model_input).cpu().detach().numpy())
    return res

  def forward(self, batch:HypothesisTensors)->torch.FloatTensor:
    assert self._is_forward_ready, "Must run model.init()"
    # seq X batch X dim
    stacked_embeddings = torch.cat([
      batch.subject_embedding.unsqueeze(0),
      batch.object_embedding.unsqueeze(0),
      batch.subject_neighbor_embeddings,
      batch.object_neighbor_embeddings,
    ])
    local_stacked_emb = self.embedding_transformation(stacked_embeddings)
    local_stacked_emb = torch.relu(local_stacked_emb)
    encoded_predicate = self.encode_predicate_data(local_stacked_emb)
    encoded_predicate = encoded_predicate.mean(dim=0)
    logit = self.encoding_to_logit(encoded_predicate)
    logit = torch.sigmoid(logit)
    return logit.reshape(-1)

  @staticmethod
  def _to_labels_n_scores(
      predictions:torch.FloatTensor,
      label:int
  )->List[Tuple[float, int]]:
    return [
        (label, score)
        for score in predictions.detach().cpu().numpy()
    ]

  def training_step(self, batch:List[HypothesisTensors], batch_idx):
    pos_batch = batch[0]
    neg_batches = batch[1:]
    pos_predictions = self.forward(self._tensors_to_device(pos_batch))
    partial_losses = []
    correctly_sorted = pos_predictions.new_zeros(1)
    incorrectly_sorted = pos_predictions.new_zeros(1)
    labels_n_scores = self._to_labels_n_scores(pos_predictions, 1)
    for idx, neg_batch in enumerate(neg_batches):
      neg_predictions = self.forward(self._tensors_to_device(neg_batch))
      labels_n_scores += self._to_labels_n_scores(neg_predictions, 0)
      partial_losses.append(self.loss_fn(
          pos_predictions,
          neg_predictions,
          pos_predictions.new_ones(len(pos_predictions))
      ))
      correctly_sorted += (
          (pos_predictions > neg_predictions)
          .detach().cpu().sum().float()
      )
      incorrectly_sorted += (
          (neg_predictions > pos_predictions)
          .detach().cpu().sum().float()
      )

    labels, scores = zip(*labels_n_scores)
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    metrics=dict(
        loss=sum(partial_losses),
        correct_comp=(
          correctly_sorted / (correctly_sorted + incorrectly_sorted)
        ),
        pr_auc=torch.tensor(pr_auc),
        roc_auc=torch.tensor(roc_auc),
    )
    for k in metrics:
      if not torch.isfinite(metrics[k]):
        metrics[k] = torch.zeros(1)
    return metrics

  def validation_step(self, batch, batch_idx):
    metrics = self.training_step(batch, batch_idx)
    metrics["val_loss"] = metrics["loss"]
    return metrics

  def _on_end(self, outputs):
    metrics = {}
    for metric in outputs[0]:
      metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()
    return metrics

  def validation_end(self, outputs):
    return self._on_end(outputs)

  def _predicate_collate(self, positive_samples):
    return predicate_collate(
        positive_samples,
        neg_scrambles_per=self.hparams.neg_scramble_rate,
        neg_swaps_per=self.hparams.neg_swap_rate,
        neighbors_per_term=self.hparams.neighbors_per_term,
        graph_index=self.graph_index,
        embedding_index=self.embedding_index,
    )

  def _config_dl(self, dataset):
    shuffle=True
    sampler=None
    if self.hparams.distributed:
      shuffle=False
      sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset=self.training_data,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=self.hparams.positives_per_batch,
        collate_fn=self._predicate_collate,
        pin_memory=True,
        #num_workers=2,
    )

  def train_dataloader(self):
    return self._config_dl(self.training_data)

  def val_dataloader(self):
    return self._config_dl(self.val_data)


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


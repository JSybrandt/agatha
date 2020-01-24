import torch
import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
from pathlib import Path
from pymoliere.ml.point_cloud_evaluator.dataset import (
    PointCloudTensors,
    PointCloudDataset,
    collate_point_clouds,
)
from pymoliere.ml.util.embedding_index import PreloadedEmbeddingIndex
from pymoliere.util.sqlite3_graph import PreloadedSqlite3Graph, Sqlite3Graph
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from typing import List, Tuple
from pymoliere.ml.abstract_generator.lamb_optimizer import Lamb

class PointCloudEvaluator(pl.LightningModule):

  def __init__(self, hparams:Namespace):
    super(PointCloudEvaluator, self).__init__()
    self.hparams = hparams

    # data indices
    embedding_index = PreloadedEmbeddingIndex(
        embedding_dir=self.hparams.embedding_dir,
        entity_dir=self.hparams.entity_dir,
        entity_types="l"
    )

    graph_index = PreloadedSqlite3Graph(self.hparams.sqlite_graph).__enter__()

    # Data set
    all_sentences = PointCloudDataset(
      entity_dir=self.hparams.entity_dir,
      embedding_index=embedding_index,
      graph_index=graph_index,
      embedding_dim=self.hparams.dim,
      source_type="s",
      neigh_type="l",
      max_neighbors=self.hparams.neighborhood_size,
    )

    # Split validation and training
    val_size = int(len(all_sentences)*self.hparams.validation_fraction)
    train_size = len(all_sentences) - val_size
    self.training_data, self.val_data = torch.utils.data.random_split(
        all_sentences, [train_size, val_size]
    )

    # Layers
    self.transform_embeddings = torch.nn.Linear(
        self.hparams.dim, self.hparams.dim
    )
    self.encoder = torch.nn.TransformerEncoder(
        encoder_layer = torch.nn.TransformerEncoderLayer(
          d_model=self.hparams.dim,
          nhead=self.hparams.transformer_heads,
          dim_feedforward=self.hparams.transformer_ff_dim,
          dropout=self.hparams.transformer_dropout,
          activation="relu",
        ),
        num_layers=self.hparams.transformer_layers,
        norm=torch.nn.LayerNorm(self.hparams.dim),
    )
    self.to_logit = torch.nn.Linear(self.hparams.dim, 1)
    # extra
    self.loss_fn = torch.nn.MarginRankingLoss(margin=0.1)

  def _tensors_to_device(self, b:PointCloudTensors)->PointCloudTensors:
    device = next(self.parameters()).device
    return PointCloudTensors(
        lemmas=b.lemmas.to(device),
    )

  def forward(self, point_clouds:PointCloudTensors)->torch.FloatTensor:
    local_lemmas = self.transform_embeddings(point_clouds.lemmas)
    local_lemmas = torch.relu(local_lemmas)
    encoded_sentence = self.encoder(local_lemmas)
    encoded_sentence = encoded_sentence.mean(dim=0)
    logit = self.to_logit(encoded_sentence)
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

  def training_step(self, batch:List[PointCloudTensors], batch_idx):
    pos_batch = batch[0]
    neg_batches = batch[1:]
    pos_predictions = self.forward(self._tensors_to_device(pos_batch))
    partial_losses = []
    labels_n_scores = self._to_labels_n_scores(pos_predictions, 1)
    for idx, neg_batch in enumerate(neg_batches):
      neg_predictions = self.forward(self._tensors_to_device(neg_batch))
      labels_n_scores += self._to_labels_n_scores(neg_predictions, 0)
      partial_losses.append(self.loss_fn(
          pos_predictions,
          neg_predictions,
          pos_predictions.new_ones(len(pos_predictions))
      ))

    labels, scores = zip(*labels_n_scores)
    metrics=dict(
        loss=sum(partial_losses),
        pr_auc=torch.tensor(average_precision_score(labels, scores)),
        roc_auc=torch.tensor(roc_auc_score(labels, scores)),
    )
    return {
        'loss': metrics["loss"],
        'progress_bar': metrics,
        'log': metrics,
    }

  def validation_step(self, batch, batch_idx):
    metrics = self.training_step(batch, batch_idx)["log"]
    metrics["val_loss"] = metrics["loss"]
    return metrics

  def _on_end(self, outputs):
    metrics = {}
    for metric in outputs[0]:
      metrics[metric] = torch.stack([x[metric] for x in outputs]).mean()
    return metrics

  def validation_end(self, outputs):
    return self._on_end(outputs)

  def _config_dl(self, dataset):
    shuffle=True
    sampler=None
    if self.hparams.distributed:
      shuffle=False
      sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    collate = lambda batch: collate_point_clouds(
        positive_examples=batch,
        neg_scrambles_per=self.hparams.neg_scramble_rate,
    )
    return torch.utils.data.DataLoader(
        dataset=self.training_data,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=self.hparams.positives_per_batch,
        collate_fn=collate,
        #num_workers=4,
    )

  @pl.data_loader
  def train_dataloader(self):
    return self._config_dl(self.training_data)

  @pl.data_loader
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
        "--sqlite-graph",
        help="Location of the graph db containing nodes and neighbors",
    )
    parser.add_argument(
        "--sqlite-embedding-location",
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
        "--warmup-steps",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--positives-per-batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=10,
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
        "--transformer-layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--transformer-ff-dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--neg-scramble-rate",
        type=int,
        default=10,
        help="A negative scramble draws the cloud sets randomly from batch"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    return parser


  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size
    )

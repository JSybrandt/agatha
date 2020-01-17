import torch
import pytorch_lightning as pl
from argparse import Namespace, ArgumentParser
from pathlib import Path
from pymoliere.ml.point_cloud_evaluator import dataset
from pymoliere.ml.util.embedding_index import EmbeddingIndex
from pymoliere.util.sqlite3_graph import Sqlite3Graph
from pymoliere.util import database_util as dbu
from pymoliere.ml.util.kv_store_dataset import KVStoreDictDataset

class PointCloudEvaluator(pl.LightningModule):

  def __init__(self, hparams:Namespace):
    super(PointCloudEvaluator, self).__init__()
    self.hparams = hparams
    self.transformer = torch.nn.Transformer(
        d_model=self.hparams.dim,
        nhead=self.hparams.transformer_heads,
        num_encoder_layers=0,
        num_decoder_layers=self.hparams.transformer_layers,
        dim_feedforward=self.hparams.dim*4,
        dropout=self.hparams.transformer_dropout,
    )
    self.dense_out = torch.nn.Linear(self.hparams.dim, 1)
    self.sig = torch.nn.Sigmoid()
    self.loss_fn = torch.nn.BCELoss()
    self.root_embedding = torch.nn.Parameter(torch.rand(1, self.hparams.dim))

  def forward(self, point_clouds:torch.FloatTensor):
    # sequence length X batch_size X dim
    assert len(point_clouds.shape) == 3
    assert point_clouds.shape[2] == self.hparams.dim
    batch_size = point_clouds.shape[1]

    # now its seq_len + 1 X batch_size X dim
    with_root = torch.cat((
      torch.stack([self.root_embedding]*batch_size, dim=1),
      point_clouds,
    ))
    # seqlen X batch size. True if padding
    padding_mask = (with_root != 0).sum(2) == 0

    encoded_result = self.transformer(
        src=with_root.new_zeros(with_root.shape),
        tgt=with_root,
        tgt_key_padding_mask=padding_mask.t_(),
    )
    # Get first embedding per batch
    point_key = encoded_result[0, :, :]
    return self.sig(self.dense_out(point_key)).reshape(-1)

  def training_step(self, batch, batch_idx):
    predictions = self.forward(batch["point_clouds"])
    labels = batch["labels"]
    if self.hparams.debug:
      print(torch.stack((predictions, labels)).detach())
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
    # point_cloud_ds = dataset.PointCloudDataset(
        # embedding_dim=self.hparams.dim,
        # entity_dir=self.hparams.entity_dir,
        # embedding_index=self.embedding_index,
        # graph_index=self.graph_index,
        # source_node_type=dbu.SENTENCE_TYPE,
        # neighbor_cloud_type=dbu.LEMMA_TYPE,
    # )
    point_cloud_ds = KVStoreDictDataset(self.hparams.training_data_dir)
    sampler=torch.utils.data.distributed.DistributedSampler(point_cloud_ds)
    return torch.utils.data.DataLoader(
        dataset=point_cloud_ds,
        sampler=sampler,
        batch_size=int(self.hparams.batch_size/2),
        # Collate will double the batch
        collate_fn=dataset.point_cloud_training_collate,
    )

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

  @staticmethod
  def configure_argument_parser(parser:ArgumentParser)->ArgumentParser:
    parser.add_argument(
        "--sqlite-graph",
        type=Path,
        help="Location of the graph db containing nodes and neighbors",
    )
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
        "--transformer-layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path
    )
    parser.add_argument("--debug", action="store_true")
    return parser


  def init_ddp_connection(self, proc_rank, world_size):
    torch.distributed.init_process_group(
        'gloo',
        rank=proc_rank,
        world_size=world_size
    )

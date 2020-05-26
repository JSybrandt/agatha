import torch
import pytorch_lightning as pl
from agatha.ml.util import hparam_util
from pytorch_lightning import Trainer
from argparse import ArgumentParser, Namespace
from typing import List, Any, Tuple, Dict
import os

class AgathaModule(pl.LightningModule):
  """
  This overrides the defaults in pytorch-lightning.

  Because we do not have a SLURM cluster, and have to modify distributed
  parameters around custom setups on Palmetto, we need some custom defaults to
  make our lives easier.

  """
  def __init__(self, hparams:Namespace):
    super(AgathaModule, self).__init__()

    # Clear paths, don't want to serialize them later
    self.hparams = hparam_util.remove_paths_from_namespace(hparams)

    # Set when init_process_group is called
    self._distributed = False

    # Set when training started
    self._training_started = False

  def set_verbose(self, val:bool)->None:
    self.hparams.verbose = val

  def _vprint(self, *args, **kwargs):
    if self.hparams.verbose:
      print(*args, **kwargs)

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

    if self.hparams.num_nodes <= 1:
      print("Number of nodes set to 1. Ignoring environment variables.")
      print("MASTER_ADDR = 127.0.0.1")
      os.environ["MASTER_ADDR"] = '127.0.0.1'
      print("MASTER_PORT = 12910")
      os.environ["MASTER_PORT"] = '12910'
    else:
      if 'MASTER_ADDR' not in os.environ:
        print("MASTER_ADDR environment variable missing. Set as localhost")
        os.environ['MASTER_ADDR'] = '127.0.0.1'

      if 'MASTER_PORT' not in os.environ:
        print("MASTER_PORT environment variable is not defined. Set as 12910")
        os.environ['MASTER_PORT'] = '12910'

    # Reverting to GLOO until nccl upgrades in pytorch are complete
    #torch_backend = "nccl" if self.trainer.on_gpu else "gloo"
    torch_backend = "gloo"

    self._vprint(
        "Attempting connection:",
        torch_backend,
        os.environ["MASTER_ADDR"], proc_rank, world_size
    )
    self._distributed = True
    torch.distributed.init_process_group(
        torch_backend,
        rank=proc_rank,
        world_size=world_size
    )

  def on_train_start(self):
    pl.LightningModule.on_train_start(self)
    self._vprint("Training started")
    self._training_started = True

  def get_device(self):
    return next(self.parameters()).device

  def training_validation_split(
      self,
      data:torch.utils.data.Dataset,
  )->Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    self._vprint("Splitting train/validation")
    v_size = int(len(data) * self.hparams.validation_fraction)
    t_size = len(data) - v_size
    self._vprint("\t- Training:", t_size)
    self._vprint("\t- Validation:", v_size)
    return torch.utils.data.random_split(data, [t_size, v_size])

  def _configure_dataloader(
      self,
      dataset:torch.utils.data.Dataset,
      batch_size:int,
      shuffle:bool=True,
      collate_fn:Any=None
  )->torch.utils.data.DataLoader:
    sampler = None
    if self._distributed:
      shuffle = False
      sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=self.hparams.dataloader_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

  def _on_epoch_end(
      self,
      outputs:List[Dict[str,torch.Tensor]]
  )->Dict[str, Dict[str,torch.Tensor]]:
    keys = outputs[0].keys()
    return {
        key: torch.mean(torch.stack([o[key] for o in outputs]))
        for key in keys
    }

  def validation_epoch_end(self, outputs:List[Dict[str,torch.Tensor]]):
    return self._on_epoch_end(outputs)

  @staticmethod
  def add_argparse_args(parser:ArgumentParser)->ArgumentParser:
    """Used to add all model parameters to argparse

    This static function allows for the easy configuration of argparse for the
    construction and training of the Agatha deep learning model. Example usage:

    ```python3
    parser = Module.add_argparse_args(ArgumentParser())
    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(args)
    model = Module(args)
    ```

    Note, many of the arguments, such as the location of training databases or
    the paths used to save the model during training, will _NOT_ be serialized
    with the model. These can be configured either from `args` directly after
    parsing, or through `configure_paths` after training.

    Args:
      parser: An argparse parser to be configured. Will receive all necessary
        training and model parameter flags.

    Returns:
      A reference to the input argument parser.

    """
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    return parser

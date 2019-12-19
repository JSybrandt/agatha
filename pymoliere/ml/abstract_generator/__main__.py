import dask
from dask.distributed import Client
import dask.bag as dbag
from nltk.tokenize import sent_tokenize
from pathlib import Path
import pickle
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.construct import dask_checkpoint, file_util, text_util, ftp_util
from pymoliere.ml.model_summary import print_model_summary
from pymoliere.ml.abstract_generator.misc_util import HashedIndex, OrderedIndex
from pymoliere.ml.abstract_generator.path_util import get_paths
from pymoliere.ml.abstract_generator import generation_util
from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
from pymoliere.ml.abstract_generator.prep_training_data import prep
from pymoliere.util.misc_util import Record, iter_to_batches
import sentencepiece as spm
import sys
import torch
from typing import Iterable, List, Dict
import random
import os
from tqdm import tqdm
import json
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sqlitedict import SqliteDict
from argparse import Namespace



# Eval added as an alias for evaluate
MODES = ["train", "evaluate", "prep", "eval"]


def get_tokenizer_from_config(
    config:cpb.AbstractGeneratorConfig
)->AbstractGeneratorTokenizer:
  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]
  assert tokenizer_model_path.is_file()
  assert extra_data_path.is_file()
  return AbstractGeneratorTokenizer(
      tokenizer_model_path=tokenizer_model_path,
      extra_data_path=extra_data_path,
  )

def get_model_from_config(
    config:cpb.AbstractGeneratorConfig,
)->AbstractGenerator:
  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]

  if config.HasField("restore_from_checkpoint"):
    return AbstractGenerator.load_from_checkpoint(config.restore_from_checkpoint)
  else:
    return AbstractGenerator(Namespace(
        tokenizer_kwargs=dict(
          tokenizer_model_path=tokenizer_model_path,
          extra_data_path=extra_data_path,
        ),
        embedding_dim=config.embedding_dim,
        max_text_length=config.text_length,
        num_attention_heads=config.num_attention_heads,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        intermediate_dropout=0.1,
        intermediate_feedforward_dim=config.hidden_fc_size,
        training_data_dir=paths["training_db_dir"],
        batch_size=config.sys.batch_size,
        warmup_steps=config.num_warmup_steps,
        learning_rate=config.sys.learning_rate,
        dataset_workers=4,
    ))

def get_device(config:cpb.AbstractGeneratorConfig)->torch.device:
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def evaluate(config:cpb.AbstractGeneratorConfig):
  if not config.HasField("restore_from_checkpoint"):
    print("WARNING: If you don't set restore_from_checkpoint,",
          "we're going to generate randomly.")
  paths = get_paths(config)

  testing_data_dir = paths["model_ckpt_dir"].joinpath("testing_data")
  assert testing_data_dir.is_dir()

  device = get_device(config)
  tokenizer = get_tokenizer_from_config(config)
  model = get_model_from_config(config).cuda()
  model.freeze()
  model.eval()

  for test_pkl in testing_data_dir.glob("*.pkl"):
    with open(test_pkl, "rb") as pkl_file:
      for record in pickle.load(pkl_file):
        print("Evaluating", record["pmid"])
        metrics = generation_util.evaluate_model_on_abstract(
            abstract=record,
            tokenizer=tokenizer,
            model=model,
            text_length=config.text_length,
            device=device,
            lowercase=config.lowercase,
        )
        print(metrics)
        # if config.HasField("eval_result_path"):
          # with open(config.eval_result_path, 'a') as out_file:
            # out_file.write(f"{json.dumps(metrics)}\n")

def distribute_training_partitions(
    partition_files:List[Path],
    rank:int,
    size:int,
    max_result_size:int,
)->List[Dict[str, torch.Tensor]]:
  print(f"Splitting {len(partition_files)} paths across {size} machines")
  # everyone needs to setup the index tensor
  indices = torch.randperm(len(partition_files))
  # reset everyone to the tensor owned by rank 0
  #indices = hvd.broadcast(indices, root_rank=0, name="indices").tolist()
  # split the indies list up
  indices = split_list_by_rank(indices, rank, size)
  #print(f"I'm responsible for {len(indices)} files:", indices)
  res = []
  for idx in indices:
    with open(partition_files[idx], 'rb') as f:
      res += pickle.load(f)
  #if max_result_size / len(res) < 0.75:
    #print(f"Warning, only selecting {max_result_size} out of {len(res)}")
  random.shuffle(res)
  return res[:max_result_size]

def train(config:cpb.AbstractGeneratorConfig):
  paths = get_paths(config)
  tokenizer = get_tokenizer_from_config(config)
  model = get_model_from_config(config)

  if config.debug:
    print_model_summary(model)

  print("Configuring trainer")
  logger = TestTubeLogger(
      save_dir=paths['model_root_dir'],
      version=config.checkpoint_version,
  )
  # DEFAULTS used by the Trainer
  checkpoint_callback = ModelCheckpoint(
    filepath=paths["model_root_dir"],
    save_best_only=False,
    verbose=True,
    monitor='loss',
    mode='min',
    prefix=''
  )

  trainer = Trainer(
      logger=logger,
      fast_dev_run=config.debug,
      gradient_clip_val=config.gradient_clip_val,
      default_save_path=paths['model_root_dir'],
      gpus=-1,
      nb_gpu_nodes=config.num_nodes if config.HasField("num_nodes") else 1,
      distributed_backend='ddp',
      accumulate_grad_batches=config.accumulate_batches,
      early_stop_callback=None,
      train_percent_check=config.training_fraction,
      checkpoint_callback=checkpoint_callback,
  )
  print("Training!")
  trainer.fit(model)




if __name__ == "__main__":
  config = cpb.AbstractGeneratorConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)

  assert config.mode in MODES
  if config.mode == "prep":
    prep(config)
  if config.mode == "train":
    train(config)
  if config.mode in {"evaluate", "eval"}:
    evaluate(config)

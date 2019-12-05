import dask
from dask.distributed import Client
import dask.bag as dbag
from nltk.tokenize import sent_tokenize
from pathlib import Path
import pickle
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.construct import dask_checkpoint, file_util, text_util, ftp_util
from pymoliere.ml.abstract_generator.misc_util import HashedIndex, OrderedIndex
from pymoliere.ml.abstract_generator import generation_util
from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
from pymoliere.ml.abstract_generator.batch_generator import (
    AbstractWindowGenerator
)
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
from sqlitedict import SqliteDict


# Eval added as an alias for evaluate
MODES = ["train", "evaluate", "prep", "eval"]


def items_to_hashed_index(collection:Iterable[str], max_index:int)->HashedIndex:
  res = HashedIndex(max_index=max_index)
  for elem in collection:
    res.add(elem)
  return res

def items_to_ordered_index(collection:Iterable[str])->OrderedIndex:
  res = OrderedIndex()
  for elem in collection:
    res.add(elem)
  return res

def connect_to_dask_cluster(config:cpb.AbstractGeneratorConfig)->None:
  # Potential cluster
  if config.cluster.run_locally or config.cluster.address == "localhost":
    print("Running dask on local machine!")
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
    if config.cluster.restart:
      print("\t- Restarting cluster...")
      dask_client.restart()

def get_paths(config:cpb.AbstractGeneratorConfig):
  """
  Returns all the relevant paths based on data from the config.
  """
  # Location we can find the existing data
  assert config.cluster.HasField("shared_scratch")
  scratch_root_dir = Path(config.cluster.shared_scratch)
  pmc_download_dir = scratch_root_dir.joinpath("pmc_raw")
  pmc_download_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_dir = scratch_root_dir.joinpath("dask_checkpoints")
  model_root_dir = \
      scratch_root_dir.joinpath("models").joinpath("abstract_generator")
  model_path = model_root_dir.joinpath("model.pt")
  model_ckpt_dir = model_root_dir.joinpath("dask_checkpoints")
  model_extra_data_path = model_root_dir.joinpath("extra_data.pkl")
  tokenizer_training_data_dir = \
      model_ckpt_dir.joinpath("tokenizer_training_data")
  tokenizer_model_path = model_root_dir.joinpath("tokenizer.model")
  tokenizer_vocab_path = model_root_dir.joinpath("tokenizer.vocab")

  if config.HasField("tokenizer_data_path"):
    tokenizer_model_path = Path(config.tokenizer_data_path)
  if config.HasField("extra_data_path"):
    model_extra_data_path = Path(config.extra_data_path)
  if config.HasField("model_path"):
    model_path = Path(config.model_path)

  # List of all directories
  dir_paths = [
      path for name, path in locals().items()
      if name.split("_")[-1]=="dir"
  ]
  # Make sure all dirs are present
  for dir_path in dir_paths:
    dir_path.mkdir(parents=True, exist_ok=True)

  # Return all paths, provided they end in "_dir" or "_path"
  return {
      name: path
      for name, path in locals().items()
      if name.split("_")[-1] in ["dir", "path"]
  }


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
    tokenizer:AbstractGeneratorTokenizer,
)->AbstractGenerator:
  paths = get_paths(config)
  training_data_dir = paths["model_root_dir"].joinpath("training_data")
  return AbstractGenerator(
      total_embed_size=len(tokenizer),
      vocab_size=tokenizer.vocab_size,
      padding_idx=tokenizer.padding_idx,
      vocab_start_idx=tokenizer.vocab_start_idx,
      embedding_dim=config.embedding_dim,
      max_text_length=config.text_length,
      num_attention_heads=config.num_attention_heads,
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers,
      intermediate_dropout=0.1,
      intermediate_feedforward_dim=config.hidden_fc_size,
      training_data_dir=training_data_dir,
      batch_size=config.sys.batch_size,
      warmup_steps=config.num_warmup_steps,
      learning_rate=config.sys.learning_rate,
  )

def get_device(config:cpb.AbstractGeneratorConfig)->torch.device:
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    return torch.device("cuda")
  else:
    return torch.device("cpu")


def evaluate(config:cpb.AbstractGeneratorConfig):
  paths = get_paths(config)

  testing_data_dir = paths["model_ckpt_dir"].joinpath("testing_data")
  assert testing_data_dir.is_dir()

  device = get_device(config)
  tokenizer = get_tokenizer_from_config(config)
  model = get_model_from_config(config, tokenizer)

  loaded_data = torch.load(paths["model_path"])
  # we want to be able to evaluate EITHER the checkpoint or final model
  if "model_state_dict" in loaded_data:
    model.load_state_dict(loaded_data["model_state_dict"])
  else:
    model.load_state_dict(loaded_data)
  model = model.eval()
  model.to(device)

  testing_data = split_partitions_across_ranks(
      testing_data_dir,
      rank=0,
      size=10,
  )
  random.shuffle(testing_data)

  with torch.no_grad():
    for record in testing_data:
      try:
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
        if config.HasField("eval_result_path"):
          with open(config.eval_result_path, 'a') as out_file:
            out_file.write(f"{json.dumps(metrics)}\n")
      except:
        print("Error evaluating", record)

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
  model = get_model_from_config(config, tokenizer)

  logger = TestTubeLogger(
      save_dir=paths['model_root_dir'],
      version=1,
    )
  trainer = Trainer(
      fast_dev_run=config.debug,
      gradient_clip_val=1,
      default_save_path=paths['model_root_dir'],
      weights_summary='full',
      gpus=-1,
      nb_gpu_nodes=4,
      distributed_backend='ddp',
      accumulate_grad_batches=4,
      # print_nan_grads=True,
      # track_grad_norm=2,
      amp_level='O3',
      use_amp=True
  )
  trainer.fit(model)


def prep(config:cpb.AbstractGeneratorConfig):
  # all important paths
  paths = get_paths(config)
  # print("Downloading PMC")
  # with ftp_util.ftp_connect(
      # address="ftp.ncbi.nlm.nih.gov",
      # workdir="/pub/pmc/oa_bulk/",
  # ) as conn:
    # xml_paths = ftp_util.ftp_retreive_all(
        # conn=conn,
        # pattern="^.*\.xml\.tar\.gz$",
        # directory=paths["pmc_download_dir"],
        # show_progress=True,
    # )
  connect_to_dask_cluster(config)
  def ckpt(val, name, overwrite=False):
    print("Checkpoint", name)
    return dask_checkpoint.checkpoint(
        val,
        name=name,
        checkpoint_dir=paths["model_ckpt_dir"],
        overwrite=overwrite,
    )


  # Get the full set of abstracts
  abstracts = file_util.load(
      paths["checkpoint_dir"]
      .joinpath("medline_documents")
  )

  interesting_abstracts = (
      abstracts
      # don't want the ones that are title-only
      .filter(lambda rec: len(rec["text_data"]) > 1)
  )
  interesting_abstracts = ckpt(interesting_abstracts, "interesting_abstracts")

  is_test_data = (
      interesting_abstracts
      .map(lambda rec: (random.random() <= config.sys.test_ratio, rec))
  )
  is_test_data = ckpt(is_test_data, "is_test_data")

  testing_data = (
      is_test_data
      .filter(lambda b_r: b_r[0])
      .map(lambda b_r: b_r[1])
  )
  testing_data = ckpt(testing_data, "testing_data")

  training_data = (
      is_test_data
      .filter(lambda b_r: not b_r[0])
      .map(lambda b_r: b_r[1])
  )
  training_data = ckpt(training_data, "training_data")

  # print("Collecting all mesh headings")
  all_mesh_headings = (
      training_data
      .map(lambda rec: rec["mesh_headings"])
      .flatten()
      .frequencies()
      .filter(lambda mesh_freq: mesh_freq[1] >= config.min_mesh_term_support)
      .map(lambda mesh_freq: mesh_freq[0])
      .compute()
  )
  print(f"Indexing all {len(all_mesh_headings)} mesh headings")
  mesh_index = items_to_ordered_index(all_mesh_headings)

  ###

  print("Getting oldest year")
  oldest_year = (
      training_data
      .filter(lambda rec: rec["date"] is not None)
      .map(lambda rec: int(rec["date"].split("-")[0]))
      .filter(lambda year: year > 1000)  # some invalid years are crazy
      .min()
      .compute()
  )
  print("\t-", oldest_year)

  ###

  print("Collecting training data for tokenizer")
  training_data_files = (
      training_data
      # Only collect 30% of abstracts
      .random_sample(0.3)
      .map_partitions(text_util.split_sentences)
      # Only need the text. We are doing a case-insensitive model.
      .map(lambda rec: rec["sent_text"])
      .map(lambda text: text.lower() if config.lowercase else text)
      # Only take 10% of sentences, ultimately,'re subsetting again
      .random_sample(0.1)
      # Reduce the total number of files
      .repartition(20)
      # Store results in textfiles
      .to_textfiles(f"{paths['tokenizer_training_data_dir']}/*.txt")
  )
  print("Training tokenizer")
  # need to place files in tokenizer_model_path
  spm.SentencePieceTrainer.train(
      f"--input={','.join(training_data_files)} "
      f"--model_prefix={paths['tokenizer_model_path'].parent}/tokenizer "
      f"--vocab_size={config.vocab_size} "
      f"--character_coverage=1.0 "
      f"--model_type=unigram "
      f"--input_sentence_size={config.max_tokenizer_sentences} "
      f"--shuffle_input_sentence=true "
  )
  assert paths["tokenizer_model_path"].is_file()
  assert paths["tokenizer_vocab_path"].is_file()

  extra_data = {
      "mesh_index": mesh_index,
      "oldest_year": oldest_year,
  }
  with open(paths["model_extra_data_path"], 'wb') as f:
    pickle.dump(extra_data, f)
  print("\t- Written:", paths["model_extra_data_path"])

  def write_windows_from_abstract(records, part_idx, database_dir):
    tokenizer = AbstractGeneratorTokenizer(
      tokenizer_model_path = paths["tokenizer_model_path"],
      extra_data_path = paths["model_extra_data_path"],
    )
    generator = AbstractWindowGenerator(
      tokenizer=tokenizer,
      records=list(records),
      batch_size=config.sys.batch_size,
      text_size=config.text_length,
      lowercase=config.lowercase,
    )
    db_path = database_dir.joinpath(f"part-{part_idx}.sqlite")
    try:
      with SqliteDict(
          db_path,
          journal_mode="OFF",
          flag="n", # create new DB
      ) as db:
        print("Writing", db_path)
        for idx, val in enumerate(generator.iterate_data_across_abstracts()):
          db[str(idx)] = val
        db.commit()
    except:
      print("Something went wrong with", db_path)
    return [1]


  # Write everything as a sqlitedb
  print("Generating windows and writing results to databases.")
  database_dir = paths["model_root_dir"].joinpath("training_data")
  database_dir.mkdir(parents=True, exist_ok=True)
  write_tasks = []
  for part_idx, part in enumerate(training_data.to_delayed()):
    write_tasks.append(
        dask.delayed(
          write_windows_from_abstract
        )(
          part, part_idx, database_dir
        )
    )
  dbag.from_delayed(write_tasks).compute()




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

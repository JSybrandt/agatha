from collections import OrderedDict
import dask
from dask.distributed import Client
from datetime import datetime
import horovod.torch as hvd
import numpy as np
from pathlib import Path
import pickle
from pprint import pprint
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.construct import dask_checkpoint, file_util, text_util
from pymoliere.ml.abstract_generator.abstract_generator import (
    INTERESTING_SENTENCE_LABLES,
    AbstractGeneratorTokenizer,
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.batch_generator import (
    AbstractWindowGenerator
)
from pymoliere.ml.train_model import train_model, split_partitions_across_ranks
from pymoliere.util.misc_util import iter_to_batches, Record
from pymongo import MongoClient
import sentencepiece as spm
import sys
import torch
from typing import Iterable
from random import random


MODES = ["train", "evaluate", "prep"]


def index_items(collection:Iterable[str])->OrderedDict:
  """
  Loop through the items and place all in an ordered dict. This is supposed to
  provide lookup tables for categorical data.
  """
  res = OrderedDict()
  for idx, item in enumerate(collection):
    res[item] = idx
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
  checkpoint_dir = scratch_root_dir.joinpath("dask_checkpoints")
  model_root_dir = scratch_root_dir.joinpath("models").joinpath("abstract_generator")
  if config.HasField("model_path"):
    model_path = Path(config.model_path)
  else:
    model_path = model_root_dir.joinpath("model.pt")
  model_ckpt_dir = model_root_dir.joinpath("dask_checkpoints")
  model_extra_data_path = model_root_dir.joinpath("extra_data.pkl")
  tokenizer_training_data_dir = \
      model_ckpt_dir.joinpath("tokenizer_training_data")
  tokenizer_model_path = model_root_dir.joinpath("tokenizer.model")
  tokenizer_vocab_path = model_root_dir.joinpath("tokenizer.vocab")

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

def evaluate(config:cpb.AbstractGeneratorConfig):
  pass

def train(config:cpb.AbstractGeneratorConfig):
  seed = 42
  hvd.init()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.set_num_threads(1)
  torch.cuda.set_device(hvd.local_rank())

  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]
  training_data_dir = paths["model_ckpt_dir"].joinpath("training_data")

  assert tokenizer_model_path.is_file()
  assert extra_data_path.is_file()
  assert training_data_dir.is_dir()

  tokenizer = AbstractGeneratorTokenizer(
      tokenizer_model_path=tokenizer_model_path,
      extra_data_path=extra_data_path,
  )

  model = AbstractGenerator(
      embedding_size=len(tokenizer),
      embedding_dim=config.embedding_dim,
      max_text_length=max(
        config.max_seed_text_length,
        config.max_follow_text_length
      ),
      num_attention_heads=config.num_attention_heads,
      num_encoder_layers=6,
      num_decoder_layers=6,
      intermediate_dropout=0.1,
      intermediate_feedforward_dim=2048,
  )

  records = split_partitions_across_ranks(
    training_data_dir,
    rank=hvd.rank(),
    size=hvd.size(),
  )

  generator = AbstractWindowGenerator(
      tokenizer=tokenizer,
      records=records,
      batch_size=config.sys.batch_size,
      required_author_count=config.max_author_count,
      required_mesh_count=config.max_mesh_count,
      seed_text_size=config.max_seed_text_length,
      follow_text_size=config.max_follow_text_length,
  )

  for seed, follow in generator:
    print(seed, follow)
    print(model(seed, follow))
    break


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
  if config.mode == "evaluate":
    evaluate(config)

  exit(0)

  ##############################################################################

  # # Prep scratches
  # scratch_root_dir = Path(config.cluster.scratch_root_dir)
  # # Used to load the sentence embedding data produced by pymoliere.construct
  # default_ckpt_dir = (
      # scratch_root_dir
      # .joinpath("dask_checkpoints")
  # )
  # if config.HasField("model_path"):
    # model_path = Path(config.model_path)
  # else:
    # model_path = (
        # scratch_root_dir
        # .joinpath("models")
        # .joinpath(MODEL_NAME)
        # .joinpath("model.pt")
    # )
  # # We're going to store model-specific checkpoints separately
  # model_ckpt_dir = (
      # scratch_root_dir
      # .joinpath("models")
      # .joinpath(MODEL_NAME)
      # .joinpath("dask_checkpoints")
  # )

  # seed = 42
  # hvd.init()

  # # We only want to do prep on the first machine
  # if hvd.rank() == 0:
    # print("Running pymoliere abstract_generator with the following parameters:")
    # print(config)


    # # Need to make sure model_path is writable
    # model_path.parent.mkdir(parents=True, exist_ok=True)
    # model_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # # All data, this is the checkpoint we depend on
    # sentences = file_util.load(
        # default_ckpt_dir.joinpath("sentences")
    # )
    # # Gets all data, returns a list of 2d arrays (sentences x embedding)
    # sentence_pairs = sentences.map_partitions(
        # util.group_sentences_into_pairs
    # )
    # print("Checkpoint: sentence_pairs")
    # checkpoint(
        # sentence_pairs,
        # name="sentence_pairs",
        # checkpoint_dir=model_ckpt_dir,
    # )

    # validation_pairs = sentence_pairs.random_sample(0.001)
    # print("Checkpoint: validation_pairs")
    # checkpoint(
        # validation_pairs,
        # name="validation_pairs",
        # checkpoint_dir=model_ckpt_dir,
    # )

  # ##############################################################################
  # seed = 42
  # hvd.init()

  # torch.manual_seed(seed)
  # torch.cuda.manual_seed(seed)
  # torch.set_num_threads(1)
  # torch.cuda.set_device(hvd.local_rank())

  # # Training data is ready, time to go!
  # if hvd.rank() == 0:
    # print("Prepping model")

  # if torch.cuda.is_available() and not config.sys.disable_gpu:
    # device = torch.device("cuda")
  # else:
    # device = torch.device("cpu")

  # model = util.AbstractGenerator.from_pretrained(
      # config.parser.bert_model,
      # freeze_bert_layers=True,
  # )
  # tokenizer = BertTokenizer.from_pretrained(config.parser.bert_model)

  # if torch.cuda.is_available and not config.sys.disable_gpu:
    # model = model.to(device)

  # if model_path.is_file:
    # if hvd.rank() == 0:
      # print("Recovering model from", model_path)
      # model.load_state_dict(torch.load(model_path))

  # hvd.broadcast_parameters(model.state_dict(), root_rank=0)

  # if hvd.rank() == 0:
    # print("Loading Data")
  # validation_data = split_partitions_across_ranks(
      # model_ckpt_dir.joinpath("validation_pairs"),
      # rank=hvd.rank(),
      # size=hvd.size(),
  # )

  # ##############################################################################
  # if config.mode == "evaluate":
    # if hvd.rank() == 0:
      # print(f"Initializing mongo connection")
    # data_collection = (
        # MongoClient(
          # host=config.result_db.address,
          # port=config.result_db.port
        # )[config.result_db.name]
        # .abstract_generator
    # )
    # for initial_sentence, follow_sentence in validation_data:
      # generated_sentence = util.generate_sentence(
          # sentence=initial_sentence,
          # model=model,
          # tokenizer=tokenizer,
          # max_sequence_length=config.parser.max_sequence_length,
          # reference_result_sentence=follow_sentence,
      # )
      # result_data = util.evaluate_generation(
          # initial_sentence=initial_sentence,
          # follow_sentence=follow_sentence,
          # generated_sentence=generated_sentence,
      # )
      # # Can't write np objects
      # result_data = {n: float(f) for n, f in result_data.items()}
      # result_data["initial_sentence"] = initial_sentence
      # result_data["follow_sentence"] = follow_sentence
      # result_data["generated_sentence"] = generated_sentence
      # # Useful helper info to track the model progress over time
      # result_data["date"] = datetime.today().strftime("%Y-%m-%d")
      # result_data["model_file_name"] = model_path.name
      # data_collection.insert_one(result_data)
      # print(generated_sentence)

  # ##############################################################################
  # elif config.mode == "train":
    # training_data = split_partitions_across_ranks(
        # model_ckpt_dir.joinpath("sentence_pairs"),
        # rank=hvd.rank(),
        # size=hvd.size(),
    # )

    # if rank == 0:
      # print("Preparing model")
    # loss_fn = torch.nn.NLLLoss()
    # optimizer = AdamW(
        # filter(lambda x: x.requires_grad, model.parameters()),
        # lr=0.002*hvd.size(),
        # correct_bias=False,
    # )
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # optimizer = hvd.DistributedOptimizer(
        # optimizer,
        # named_parameters=model.named_parameters(),
    # )

    # num_batches = int(config.examples_per_epoch / config.sys.batch_size)
    # num_batches=int(num_batches / hvd.size())

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        # optimizer,
        # max_lr=0.01,
        # steps_per_epoch=num_batches,
        # epochs=config.sys.num_epochs,
    # )

    # def start_epoch(epoch:int):
      # shuffle(training_data)
      # # We're going fine-tune the softmax layer in the first epoch,
      # # and then all is fair game
      # if 1 <= epoch <= 12:
        # # Epoch 0, everything is frozen. Each epoch thereafter we enable a layer.
        # model.unfreeze_layers_starting_with(12-epoch)
      # if (
          # epoch > 0
          # and epoch % 5 == 0
          # and hvd.rank() == 0
      # ):
        # print("Saving model")
        # torch.save(model.state_dict(), f"{model_path}.{epoch}")

    # def gen_batch(epoch:int):
      # # Difficulty rises from 0.1 -> 1 in half the epochs
      # mod = np.interp(
          # epoch,
          # xp=[0, config.sys.num_epochs/2, config.sys.num_epochs],
          # fp=[0.1, 1, 1]
      # )
      # for batch in iter_to_batches(training_data, config.sys.batch_size):
        # in_kwargs, expected_out = util.sentence_pairs_to_model_io(
            # tokenizer=tokenizer,
            # batch_pairs=batch,
            # # The unchanged rate drops as difficulty increases
            # unchanged_prob=config.unchanged_prob*(1-mod),
            # # Other params increase in difficulty
            # full_mask_prob=config.full_mask_prob*mod,
            # mask_per_token_prob=config.mask_per_token_prob*mod,
            # replace_per_token_prob=config.replace_per_token_prob*mod,
            # max_sequence_length=config.parser.max_sequence_length,
        # )
        # in_kwargs = {k: v.to(device) for k, v in in_kwargs.items()}
        # yield in_kwargs, expected_out.to(device)

    # def gen_validation_batch(epoch:int):
      # for batch in iter_to_batches(validation_data, config.sys.batch_size):
        # in_kwargs, expected_out = util.sentence_pairs_to_model_io(
            # tokenizer=tokenizer,
            # batch_pairs=batch,
            # # turn off everything except full mask
            # unchanged_prob=0,
            # replace_per_token_prob=0,
            # mask_per_token_prob=0,
            # full_mask_prob=1,
            # max_sequence_length=config.parser.max_sequence_length,
        # )
        # in_kwargs = {k: v.to(device) for k, v in in_kwargs.items()}
        # yield in_kwargs, expected_out.to(device)

    # #total_batches = int(len(data) / config.sys.batch_size)

    # def after_loss_calculation(loss):
      # # Only runs when phase == train
      # loss.backward()
      # # optimizer.synchronize()
      # #torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
      # # with optimizer.skip_synchronize():
      # optimizer.step()
      # scheduler.step()
      # optimizer.zero_grad()

    # def calc_accuracy(predicted, expected):
      # # predicted.shape = batch x seq_len x voccab size (float softmax)
      # # expected.shape = batch x seq_len (ints)
      # # Must produce accuracy per batch
      # # Don't want to count the padding

      # valid_mask = expected != 0
      # num_expected = valid_mask.sum().float()

      # predicted_labels = torch.argmax(predicted, dim=2)
      # assert predicted_labels.shape == expected.shape

      # num_correct = (
          # (predicted_labels[valid_mask] == expected[valid_mask])
          # .sum().float()
      # )
      # return num_correct/num_expected


    # def loss_wrapper(predicted, expected):
      # # predicted.shape = batch x seq_len x voccab size (float softmax)
      # # expected.shape = batch x seq_len (ints)
      # expanded_size = expected.shape[0] * expected.shape[1]
      # return loss_fn(
          # predicted.view(expanded_size, -1),
          # expected.view(-1),
      # )

    # def get_overall_averages_for_metrics(phase, metric2score):
      # if hvd.rank() == 0:
        # print("Metric Summary:", phase)
      # # sorted list to ensure that keys are encountered in the same order
      # for metric, score in sorted(list(metric2score.items())):
        # score = hvd.allreduce(score, name=metric)
        # if hvd.rank() == 0:
          # print(f"\t- {metric}: {score.item()}")
      # if hvd.rank() == 0:
        # print("\n\n")

    # train_model(
        # model=model,
        # loss_fn=loss_wrapper,
        # num_epochs=config.sys.num_epochs,
        # on_epoch_start=start_epoch,
        # batch_generator=gen_batch,
        # validation_batch_generator=gen_validation_batch,
        # after_loss_calculation=after_loss_calculation,
        # metrics=[
            # ("accuracy", calc_accuracy)
        # ],
        # disable_pbar=True,
        # # Turns out transmitting the plots over horovod will break the pipeline
        # disable_plots=True,
        # disable_batch_report=hvd.rank() != 0,
        # num_batches=num_batches,
        # on_phase_end=get_overall_averages_for_metrics,
    # )

    # ############################################################################

    # if hvd.rank() == 0:
      # print("Saving model")
      # torch.save(model.state_dict(), model_path)

def prep(config:cpb.AbstractGeneratorConfig):
  connect_to_dask_cluster(config)
  # all important paths
  paths = get_paths(config)
  def ckpt(val, name):
    print("Checkpoint", name)
    return dask_checkpoint.checkpoint(
        val,
        name=name,
        checkpoint_dir=paths["model_ckpt_dir"],
    )


  # Get the full set of abstracts
  abstracts = file_util.load(
      paths["checkpoint_dir"]
      .joinpath("medline_documents")
  )

  def all_text_fields_labeled(record:Record)->bool:
    for field in record["text_data"]:
      if field["type"] not in INTERESTING_SENTENCE_LABLES:
        return False
    return True

  interesting_abstracts = (
      abstracts
      .filter(lambda rec: len(rec["text_data"]) > 1)
      .filter(all_text_fields_labeled)
  )
  ckpt(interesting_abstracts, "interesting_abstracts")

  is_test_data = (
      interesting_abstracts
      .map(lambda rec: (random() <= config.sys.test_ratio, rec))
  )
  ckpt(is_test_data, "is_test_data")

  testing_data = (
      is_test_data
      .filter(lambda b_r: b_r[0])
      .map(lambda b_r: b_r[1])
  )
  ckpt(testing_data, "testing_data")

  training_data = (
      is_test_data
      .filter(lambda b_r: not b_r[0])
      .map(lambda b_r: b_r[1])
  )
  ckpt(training_data, "training_data")

  ###

  def get_authors(records:Iterable[Record])->Iterable[str]:
    res = []
    for record in records:
      res += record["authors"]
    return res
  print("Calculating frequent authors")
  # collection of (name, count) pairs
  frequent_authors = (
      training_data
      # Select all the authors
      .map_partitions(get_authors)
      # Count the number of papers per author
      .frequencies()
      # Get all the authors with more than 1 paper
      .filter(lambda a_f: a_f[1] > 1)
      # Select only the author names
      .map(lambda a_f: a_f[0])
      .compute()
  )
  print(f"\t- Found {len(frequent_authors)} frequent authors.")
  author_index = index_items(
    ["[PAD]", "[UNK]", "[MASK]"] + frequent_authors
  )
  # Add some helper tokens to the front of the list
  frequent_authors = ["[PAD]", "[UNK]", "[MASK]"] + frequent_authors
  # Using ordered dict to make it easier to go idx -> author
  author_index = OrderedDict()
  for idx, name in enumerate(frequent_authors):
    author_index[name] = idx

  ###

  def get_mesh_headings(records:Iterable[Record])->Iterable[str]:
    res = set()
    for record in records:
      for term in record["mesh_headings"]:
        res.add(term)
    return res
  print("Collecting all mesh headings")
  all_mesh_headings = (
      training_data
      .map_partitions(get_mesh_headings)
      .distinct()
      .compute()
  )
  print(f"\t- Found {len(all_mesh_headings)}")
  mesh_index = index_items(
    ["[PAD]", "[UNK]", "[MASK]"] + all_mesh_headings
  )

  ###

  print("Getting oldest year")
  oldest_year = (
      training_data
      .map(lambda rec: int(rec["date"].split("-")[0]))
      .min()
      .compute()
  )
  print("\t-", oldest_year)

  ###

  print("Collecting training data for tokenizer")
  training_data_files = (
      training_data
      # Only collect 10% of abstracts
      .random_sample(0.1)
      .map_partitions(text_util.split_sentences)
      # Only need the text. We are doing a case-insensitive model.
      .map(lambda rec: rec["sent_text"].lower())
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
      "author_index": author_index,
      "mesh_index": mesh_index,
      "oldest_year": oldest_year,
  }
  with open(paths["model_extra_data_path"], 'wb') as f:
    pickle.dump(extra_data, f)
  print("\t- Written:", paths["model_extra_data_path"])

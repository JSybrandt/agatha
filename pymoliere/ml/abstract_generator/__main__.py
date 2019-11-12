import dask
from dask.distributed import Client
import horovod.torch as hvd
from pathlib import Path
import pickle
from pymoliere.config import config_pb2 as cpb, proto_util
from pymoliere.construct import dask_checkpoint, file_util, text_util
from pymoliere.ml.model_summary import print_model_summary
from pymoliere.ml.abstract_generator.misc_util import HashedIndex
from pymoliere.ml.abstract_generator.abstract_generator import (
    INTERESTING_SENTENCE_LABLES,
    AbstractGeneratorTokenizer,
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.batch_generator import (
    AbstractWindowGenerator
)
from pymoliere.ml.train_model import train_model, split_partitions_across_ranks
from pymoliere.util.misc_util import Record
import sentencepiece as spm
import sys
import torch
from typing import Iterable
import random
import os


MODES = ["train", "evaluate", "prep"]

def index_items(collection:Iterable[str], max_index:int)->HashedIndex:
  res = HashedIndex(max_index=max_index)
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
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  seed = 42
  hvd.init()
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.set_num_threads(4)
  torch.cuda.set_device(hvd.local_rank())

  paths = get_paths(config)
  tokenizer_model_path = paths["tokenizer_model_path"]
  extra_data_path = paths["model_extra_data_path"]
  training_data_dir = paths["model_ckpt_dir"].joinpath("training_data")

  assert tokenizer_model_path.is_file()
  assert extra_data_path.is_file()
  assert training_data_dir.is_dir()

  if hvd.rank() == 0:
    print("Preparing model")
  tokenizer = AbstractGeneratorTokenizer(
      tokenizer_model_path=tokenizer_model_path,
      extra_data_path=extra_data_path,
      required_author_count=config.max_author_count,
      required_mesh_count=config.max_mesh_count,
  )

  model = AbstractGenerator(
      embedding_size=len(tokenizer),
      embedding_dim=config.embedding_dim,
      max_text_length=max(
        config.max_seed_text_length,
        config.max_follow_text_length
      ),
      num_attention_heads=config.num_attention_heads,
      num_encoder_layers=config.num_encoder_layers,
      num_decoder_layers=config.num_decoder_layers,
      intermediate_dropout=0.1,
      intermediate_feedforward_dim=config.hidden_fc_size,
      num_metadata_embeddings=tokenizer.num_metadata_embeddings()
  )
  #print_model_summary(model)
  model.to(device)

  loss_fn = torch.nn.NLLLoss()
  optimizer = torch.optim.Adam(
      model.parameters(),
      # facebook paper says linear growth with batch size
      lr=config.sys.learning_rate*hvd.size(),
  )
  # Update everybody
  hvd.broadcast_parameters(model.state_dict(), root_rank=0)
  hvd.broadcast_optimizer_state(optimizer, root_rank=0)
  # Prep for horovod
  optimizer = hvd.DistributedOptimizer(
      optimizer,
      named_parameters=model.named_parameters(),
      compression=hvd.Compression.fp16,
  )

  def loss_wrapper(predicted, expected):
    assert len(predicted.shape) == 3
    assert len(expected.shape) == 2
    # sequence length
    assert predicted.shape[0] == expected.shape[0]
    # Batch size
    assert predicted.shape[1] == expected.shape[1]
    vocab_size = predicted.shape[2]
    return loss_fn(predicted.view(-1, vocab_size), expected.view(-1))

  def on_epoch_start(epoch):
    if (
        epoch > 0
        and epoch % 5  == 0
        and hvd.rank() == 0
    ):
      print("Saving model checkpoint")
      torch.save(model.state_dict(), f"{paths['model_path']}.{epoch}")

  if hvd.rank() == 0:
    print("Loading")
  training_data = split_partitions_across_ranks(
      training_data_dir,
      rank=hvd.rank(),
      size=hvd.size(),
  )
  num_batches = int(
      config.examples_per_epoch / (config.sys.batch_size * hvd.size())
  )
  def generator_wrapper(epoch):
    random.shuffle(training_data)
    generator = AbstractWindowGenerator(
        num_workers=3,  # one worker, does it async
        queue_size=10,
        device=device,
        # Batch generator kwargs
        records=training_data,
        difficulty=0.1,
        batch_size=config.sys.batch_size,
        seed_text_size=config.max_seed_text_length,
        follow_text_size=config.max_follow_text_length,
        # tokenizer kwargs
        tokenizer_model_path=paths["tokenizer_model_path"],
        extra_data_path=paths["model_extra_data_path"],
        required_author_count=config.max_author_count,
        required_mesh_count=config.max_mesh_count,
    )
    for batch in generator.generate():
        yield batch

  def text_accuracy(predicted, expected):
    # Predicted is size (F, B, V) and expected is size (F, B)
    assert predicted.shape[0] == expected.shape[0]
    assert predicted.shape[1] == expected.shape[1]
    metadata_size = tokenizer.num_metadata_embeddings()
    assert predicted.shape[0] > metadata_size + 2
    # strip out metadata columns
    predicted = predicted[metadata_size:, :, :].argmax(dim=2)
    expected = expected[metadata_size:, :]
    mask = torch.ones_like(expected, dtype=torch.bool)
    # set to false the padding and sep tokens, leaving only the real text
    mask &= expected != tokenizer.padding_idx
    mask &= expected != tokenizer.sep_idx
    # Correct text prediction rate across batch
    return (predicted[mask] == expected[mask]).float().mean()

  def end_type_accuracy(predicted, expected):
    assert predicted.shape[0] == expected.shape[0]
    assert predicted.shape[1] == expected.shape[1]
    end_type_idx = 3 # depends on the tokenizer
    predicted = predicted[end_type_idx,:,:].argmax(dim=1)
    expected = expected[end_type_idx,:]
    return (predicted == expected).float().mean()


  print("Ready to go!")
  train_model(
      model=model,
      loss_fn=loss_wrapper,
      num_epochs=config.sys.num_epochs,
      on_epoch_start=on_epoch_start,
      batch_generator=generator_wrapper,
      optimizer=optimizer,
      disable_pbar=True,
      disable_plots=True,
      disable_batch_report=hvd.rank() != 0,
      num_batches=num_batches,
      metrics = [
        ("text_acc", text_accuracy),
        ("end_type_acc", end_type_accuracy),
      ]
  )

  if hvd.rank() == 0:
    print("Saving model")
    torch.save(model.state_dict(), paths["model_path"])


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
      .map(lambda rec: (random.random() <= config.sys.test_ratio, rec))
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
  def get_distinct_from_list(records, field):
    res = set()
    for record in records:
      for elem in record[field]:
        res.add(elem)
    return res

  ###

  print("Collecting all authors")
  all_authors = (
      training_data
      .map_partitions(get_distinct_from_list, field="authors")
      .distinct()
      .compute()
  )
  print(f"Hashing {len(all_authors)} to {config.author_hash_size}")
  author_index = index_items(all_authors, config.author_hash_size)

  print("Collecting all mesh headings")
  all_mesh_headings = (
      training_data
      .map_partitions(get_distinct_from_list, field="mesh_headings")
      .distinct()
      .compute()
  )
  print(f"Hashing {len(all_mesh_headings)} to {config.mesh_hash_size}")
  mesh_index = index_items(all_mesh_headings, config.mesh_hash_size)

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

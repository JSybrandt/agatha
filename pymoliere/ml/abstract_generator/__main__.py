from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pathlib import Path
import pymoliere.ml.abstract_generator.util as util
from pymoliere.construct import file_util
import torch
import dask
from dask.distributed import Client
from pymoliere.construct.dask_checkpoint import checkpoint
from pymoliere.ml.train_model import train_model
from sklearn.utils import shuffle
from transformers import BertTokenizer, AdamW
from pymoliere.util.misc_util import iter_to_batches
import sys


if __name__ == "__main__":
  model_name = "abstract_generator"
  config = cpb.AbstractGeneratorConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere sentence_classifier with the following parameters:")
  print(config)

  # Potential cluster
  if config.cluster.run_locally or config.cluster.address == "localhost":
    print("Running on local machine!")
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
    if config.cluster.restart:
      print("\t- Restarting cluster...")
      dask_client.restart()

  # Prep scratches
  shared_scratch = Path(config.cluster.shared_scratch)
  # Used to load the sentence embedding data produced by pymoliere.construct
  default_ckpt_dir = (
      shared_scratch
      .joinpath("dask_checkpoints")
  )
  model_path = (
      shared_scratch
      .joinpath("models")
      .joinpath(model_name)
      .joinpath("model.pt")
  )
  # Need to make sure model_path is writable
  model_path.parent.mkdir(parents=True, exist_ok=True)

  # We're going to store model-specific checkpoints separately
  data_ckpt_dir = (
      shared_scratch
      .joinpath("models")
      .joinpath(model_name)
      .joinpath("dask_checkpoints")
  )
  data_ckpt_dir.mkdir(parents=True, exist_ok=True)


  # All data, this is the checkpoint we depend on
  sentences = file_util.load(
      default_ckpt_dir.joinpath("sentences")
  )
  # Gets all data, returns a list of 2d arrays (sentences x embedding)
  sentence_pairs = sentences.map_partitions(
      util.group_sentences_into_pairs
  )
  print("Checkpoint: sentence_pairs")
  checkpoint(
      sentence_pairs,
      name="sentence_pairs",
      checkpoint_dir=data_ckpt_dir,
  )

  # Training data is ready, time to go!
  print("Prepping model")
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = util.AbstractGenerator.from_pretrained(
      config.parser.bert_model,
      freeze_bert_layers=False,
  )
  tokenizer = BertTokenizer.from_pretrained(config.parser.bert_model)

  if model_path.is_file():
    print("Loading Model")
    model.load_state_dict(torch.load(model_path))

  print("Model -> Device")
  model.to(device)

  loss_fn = torch.nn.NLLLoss()
  optimizer = AdamW(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
      correct_bias=False,
  )

  # Prep for training
  print("Loading Data")
  data = file_util.load_to_memory(
      data_ckpt_dir.joinpath("sentence_pairs"),
  )

  def start_epoch():
    print("Saving model")
    torch.save(model.state_dict(), model_path)
    shuffle(data)

  def gen_batch():
    for batch in iter_to_batches(data[:100000], config.sys.batch_size):
      yield util.sentence_pairs_to_tensor_batch(
          tokenizer=tokenizer,
          batch_pairs=batch,
          unchanged_prob=config.unchanged_prob,
          full_mask_prob=config.full_mask_prob,
          mask_per_token_prob=config.mask_per_token_prob,
          max_sequence_length=config.parser.max_sequence_length,
      )
  total_batches = int(len(data) / config.sys.batch_size)

  def after_loss_calculation(loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  def calc_accuracy(predicted, expected):
    expanded_size = expected.shape[0] * expected.shape[1]
    predicted_labels = torch.argmax(predicted.view(expanded_size, -1), dim=1)
    num_correct = (predicted_labels == expected.view(-1)).sum().float()
    return num_correct / expanded_size

  def loss_wrapper(predicted, expected):
    expanded_size = expected.shape[0] * expected.shape[1]
    return loss_fn(
        predicted.view(expanded_size, -1),
        expected.view(-1),
    )

  train_model(
      model=model,
      loss_fn=loss_wrapper,
      num_epochs=config.sys.num_epochs,
      on_epoch_start=start_epoch,
      batch_generator=gen_batch,
      after_loss_calculation=after_loss_calculation,
      num_batches=total_batches,
      metrics=[
          ("accuracy", calc_accuracy)
      ]
  )
  print("Saving model")
  torch.save(model.state_dict(), model_path)

  print("Play with the model!")
  model.eval()
  for sentence in sys.stdin:
    sentence = sentence.strip()
    for _ in range(4):
      sentence = util.generate_sentence(
          sentence=sentence,
          max_sequence_length=config.parser.max_sequence_length,
          model=model,
          tokenizer=tokenizer,
      )
      print(sentence)

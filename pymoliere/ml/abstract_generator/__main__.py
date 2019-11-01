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
from pymoliere.ml.train_model import train_model, split_partitions_across_ranks
from sklearn.utils import shuffle
from transformers import BertTokenizer, AdamW
from pymoliere.util.misc_util import iter_to_batches
import sys
import horovod.torch as hvd
import numpy as np


if __name__ == "__main__":
  model_name = "abstract_generator"
  config = cpb.AbstractGeneratorConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)

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
  # We're going to store model-specific checkpoints separately
  data_ckpt_dir = (
      shared_scratch
      .joinpath("models")
      .joinpath(model_name)
      .joinpath("dask_checkpoints")
  )

  if config.use_horovod:
    seed = 42
    hvd.init()

  # We only want to do prep on the first machine
  if not config.use_horovod or hvd.rank() == 0:
    print("Running pymoliere abstract_generator with the following parameters:")
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

    # Need to make sure model_path is writable
    model_path.parent.mkdir(parents=True, exist_ok=True)
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

    validation_pairs = sentence_pairs.random_sample(0.001)
    print("Checkpoint: validation_pairs")
    checkpoint(
        validation_pairs,
        name="validation_pairs",
        checkpoint_dir=data_ckpt_dir,
    )


  ##############################################################################

  if config.use_horovod:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)
    torch.cuda.set_device(hvd.local_rank())

  # Training data is ready, time to go!
  if not config.use_horovod or hvd.rank() == 0:
    print("Prepping model")

  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")

  else:
    device = torch.device("cpu")

  model = util.AbstractGenerator.from_pretrained(
      config.parser.bert_model,
      freeze_bert_layers=True,
  )
  tokenizer = BertTokenizer.from_pretrained(config.parser.bert_model)

  if torch.cuda.is_available and not config.sys.disable_gpu:
    model = model.to(device)

  loss_fn = torch.nn.NLLLoss()
  lr = 0.002
  if config.use_horovod:
    lr *= hvd.size()
  optimizer = AdamW(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=lr,
      correct_bias=False,
  )
  if config.use_horovod:
    if hvd.rank() == 0:
      print("Loading Data")
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    # compression = hvd.Compression.fp16
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        # compression=compression,
    )
    data = split_partitions_across_ranks(
        data_ckpt_dir.joinpath("sentence_pairs"),
        rank=hvd.rank(),
        size=hvd.size(),
    )
    validation_data = split_partitions_across_ranks(
        data_ckpt_dir.joinpath("validation_pairs"),
        rank=hvd.rank(),
        size=hvd.size(),
    )
  else:
    print("Loading all data")
    data = file_util.load_to_memory(
        data_ckpt_dir.joinpath("sentence_pairs"),
    )
    validation_data = file_util.load_to_memory(
        data_ckpt_dir.joinpath("validation_pairs"),
    )

  num_batches = int(config.examples_per_epoch / config.sys.batch_size)
  if config.use_horovod:
    num_batches=int(num_batches / hvd.size())

  scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=0.01,
      steps_per_epoch=num_batches,
      epochs=config.sys.num_epochs,
  )

  def start_epoch(epoch:int):
    shuffle(data)
    # We're going fine-tune the softmax layer in the first epoch,
    # and then all is fair game
    if 1 <= epoch <= 12:
      # Epoch 0, everything is frozen. Each epoch thereafter we enable a layer.
      model.unfreeze_layers_starting_with(12-epoch)
    if (
        epoch > 0
        and epoch % 5 == 0
        and (not config.use_horovod or hvd.rank() == 0)
    ):
      print("Saving model")
      torch.save(model.state_dict(), f"{model_path}.{epoch}")

  def gen_batch(epoch:int):
    # Difficulty rises from 0.1 -> 1 in half the epochs
    mod = np.interp(
        epoch,
        xp=[0, config.sys.num_epochs/2, config.sys.num_epochs],
        fp=[0.1, 1, 1]
    )
    for batch in iter_to_batches(data, config.sys.batch_size):
      in_kwargs, expected_out = util.sentence_pairs_to_model_io(
          tokenizer=tokenizer,
          batch_pairs=batch,
          # The unchanged rate drops as difficulty increases
          unchanged_prob=config.unchanged_prob*(1-mod),
          # Other params increase in difficulty
          full_mask_prob=config.full_mask_prob*mod,
          mask_per_token_prob=config.mask_per_token_prob*mod,
          replace_per_token_prob=config.replace_per_token_prob*mod,
          max_sequence_length=config.parser.max_sequence_length,
      )
      in_kwargs = {k: v.to(device) for k, v in in_kwargs.items()}
      yield in_kwargs, expected_out.to(device)

  def gen_validation_batch(epoch:int):
    for batch in iter_to_batches(validation_data, config.sys.batch_size):
      in_kwargs, expected_out = util.sentence_pairs_to_model_io(
          tokenizer=tokenizer,
          batch_pairs=batch,
          # turn off everything except full mask
          unchanged_prob=0,
          replace_per_token_prob=0,
          mask_per_token_prob=0,
          full_mask_prob=1,
          max_sequence_length=config.parser.max_sequence_length,
      )
      in_kwargs = {k: v.to(device) for k, v in in_kwargs.items()}
      yield in_kwargs, expected_out.to(device)

  #total_batches = int(len(data) / config.sys.batch_size)

  def after_loss_calculation(loss):
    # Only runs when phase == train
    loss.backward()
    # optimizer.synchronize()
    #torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    # with optimizer.skip_synchronize():
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  def calc_accuracy(predicted, expected):
    # predicted.shape = batch x seq_len x voccab size (float softmax)
    # expected.shape = batch x seq_len (ints)
    # Must produce accuracy per batch
    # Don't want to count the padding

    valid_mask = expected != 0
    num_expected = valid_mask.sum().float()

    predicted_labels = torch.argmax(predicted, dim=2)
    assert predicted_labels.shape == expected.shape

    num_correct = (
        (predicted_labels[valid_mask] == expected[valid_mask])
        .sum().float()
    )
    return num_correct/num_expected


  def loss_wrapper(predicted, expected):
    # predicted.shape = batch x seq_len x voccab size (float softmax)
    # expected.shape = batch x seq_len (ints)
    expanded_size = expected.shape[0] * expected.shape[1]
    return loss_fn(
        predicted.view(expanded_size, -1),
        expected.view(-1),
    )

  def get_overall_averages_for_metrics(phase, metric2score):
    if not config.use_horovod:
      print("Metric Summary")
      print(metric2score)
    else:
      if hvd.rank() == 0:
        print("Metric Summary:", phase)
      # sorted list to ensure that keys are encountered in the same order
      for metric, score in sorted(list(metric2score.items())):
        score = hvd.allreduce(score, name=metric)
        if hvd.rank() == 0:
          print(f"\t- {metric}: {score.item()}")
      if hvd.rank() == 0:
        print("\n\n")

  train_model(
      model=model,
      loss_fn=loss_wrapper,
      num_epochs=config.sys.num_epochs,
      on_epoch_start=start_epoch,
      batch_generator=gen_batch,
      validation_batch_generator=gen_validation_batch,
      after_loss_calculation=after_loss_calculation,
      metrics=[
          ("accuracy", calc_accuracy)
      ],
      disable_pbar=config.use_horovod, # Don't show pbar if distributed
      # Turns out transmitting the plots over horovod will break the pipeline :P
      disable_plots=config.use_horovod,
      disable_batch_report=(config.use_horovod and not hvd.rank() == 0),
      num_batches=num_batches,
      on_phase_end=get_overall_averages_for_metrics,
  )

  ##############################################################################

  if not config.use_horovod or hvd.rank() == 0:
    print("Saving model")
    torch.save(model.state_dict(), model_path)

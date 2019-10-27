# TRAIN SENTENCE CLASSIFIER
# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pathlib import Path
from pymoliere.construct import file_util
from pymoliere.ml.train_model import train_model, split_data_across_ranks
import torch
import dask
from dask.distributed import Client
from pymoliere.construct.dask_checkpoint import checkpoint
from pymoliere.ml.sentence_classifier.util import (
    TrainingData,
    get_boundary_dates,
    filter_sentences_with_embedding,
    record_to_training_tuple,
    SentenceClassifier,
    IDX2LABEL,
)
from pymoliere.util.misc_util import iter_to_batches
import horovod.torch as hvd


if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)

  # Paths
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
      .joinpath("sentence_classifier")
      .joinpath("model.pt")
  )
  if config.HasField("custom_data_dir"):
    data_ckpt_dir = Path(config.custom_data_dir)
    assert data_ckpt_dir.is_dir()
  else:
    data_ckpt_dir = (
        shared_scratch
        .joinpath("models")
        .joinpath("sentence_classifier")
        .joinpath("dask_checkpoints")
    )

  if config.use_horovod:
    seed = 42
    hvd.init()

  # Prep training data, use one rank to begin.
  if not config.use_horovod or hvd.rank() == 0:
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

    # Need to make sure model_path is writable
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # We're going to store model-specific checkpoints separately
    data_ckpt_dir.mkdir(parents=True, exist_ok=True)


    # All data, this is the checkpoint we depend on
    sentences_with_embedding = file_util.load(
        default_ckpt_dir.joinpath("sentences_with_embedding")
    )
    # Get only results with labels, store at TrainingData tuples
    all_data = sentences_with_embedding.map_partitions(
        filter_sentences_with_embedding
    )
    print("Checkpoint: all_data")
    checkpoint(
        all_data,
        name="all_data",
        checkpoint_dir=data_ckpt_dir,
    )

    if not file_util.is_result_saved(
        data_ckpt_dir.joinpath("training_data")
    ):
      print("Finding the training data!")
      print("Getting Dates")
      sample_of_dates = (
          all_data
          .random_sample(0.05)
          .map(lambda x: x.date)
          .compute()
      )
      val_date, test_date = get_boundary_dates(
          dates=sample_of_dates,
          validation_ratio=config.validation_set_ratio,
          test_ratio=config.test_set_ratio,
      )
      print("Training goes up to ", val_date)
      print(f"Validation is from {val_date} to {test_date}")
      print("Testing is after", test_date)
      save_training_data = file_util.save(
          bag=(
            all_data
            .filter(lambda x: x.date < val_date)
          ),
          path=data_ckpt_dir.joinpath("training_data")
      )
      save_validation_data = file_util.save(
          bag=(
            all_data
            .filter(lambda x: val_date <= x.date < test_date)
          ),
          path=data_ckpt_dir.joinpath("validation_data")
      )
      save_test_data = file_util.save(
          bag=(
            all_data
            .filter(lambda x: test_date <= x.date)
          ),
          path=data_ckpt_dir.joinpath("testing_data")
      )
      print("Filtering and saving training/validation/testing data.")
      dask.compute(save_training_data, save_validation_data, save_test_data)

  ##############################################################################

  if config.use_horovod:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_num_threads(1)
    torch.cuda.set_device(hvd.local_rank())

  # Training data is ready, time to go!
  print("Prepping model")
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier()

  print("Model -> Device")
  model.to(device)

  lr = 0.002
  if config.use_horovod:
    lr *= hvd.size()

  loss_fn = torch.nn.NLLLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=lr,
  )

  if config.use_horovod:
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    compression = hvd.Compression.fp16
    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
    )


  # Get your data!
  print("Loading training data")
  training_data = file_util.load_to_memory(
      data_ckpt_dir.joinpath("training_data"),
      disable_pbar=config.use_horovod, # Don't show pbar if distributed
  )
  validation_data = file_util.load_to_memory(
      data_ckpt_dir.joinpath("validation_data"),
      disable_pbar=config.use_horovod, # Don't show pbar if distributed
  )

  if config.use_horovod:
    split_data_across_ranks(training_data)
    split_data_across_ranks(validation_data)

  ## Helper functions for training
  def on_epoch_start(epoch):
    shuffle(training_data)
    if epoch > 0 and (not config.use_horovod or hvd.rank() == 0):
      print("Saving model")
      torch.save(model.state_dict(), model_path)

  def gen_batch(data):
    for batch in iter_to_batches(data, config.sys.batch_size):
      yield(
          {"x":torch.stack([torch.FloatTensor(b.dense_data) for b in batch]).to(device)},
          torch.LongTensor([b.label for b in batch]).to(device),
      )
  def training_data_generator():
    return gen_batch(training_data)
  def validation_data_generator():
    return gen_batch(validation_data)

  def after_loss_calculation(loss):
    optimizer.zero_grad()
    loss.backward()
    if config.use_horovod:
      optimizer.synchronize()
      torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
      with optimizer.skip_synchronize():
        optimizer.step()
    else:
      optimizer.step()

  def calc_accuracy(predicted_logits, expected):
    predicted_values = torch.argmax(predicted_logits, dim=1)
    num_correct = torch.sum(predicted_values == expected).sum().float()
    # accuracy in batch is divided by batch size
    return num_correct / predicted_logits.shape[0]

  train_model(
      model=model,
      loss_fn=loss_fn,
      num_epochs=config.sys.num_epochs,
      batch_generator=training_data_generator,
      validation_batch_generator=validation_data_generator,
      after_loss_calculation=after_loss_calculation,
      metrics=[
          ("accuracy", calc_accuracy)
      ],
      disable_pbar=config.use_horovod, # Don't show pbar if distributed
      # Turns out transmitting the plots over horovod will break the pipeline :P
      disable_plots=config.use_horovod,
  )

  if not config.use_horovod or hvd.rank() == 0:
    print("Saving model")
    torch.save(model.state_dict(), model_path)



  # print(f"Loading test data from {data_ckpt_dir}")
  # test_data = file_util.load_to_memory(
      # data_ckpt_dir.joinpath("testing_data")
  # )

  # print("Evaluation")
  # evaluate_model.evaluate_multiclass_model(
      # model=model,
      # device=device,
      # batch_size=config.sys.batch_size,
      # data=[x.dense_data for x in test_data],
      # labels=[x.label for x in test_data],
      # class_names=IDX2LABEL,
  # )


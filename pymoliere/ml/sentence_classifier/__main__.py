# TRAIN SENTENCE CLASSIFIER
# Train a sentence classifier to apply to pymoliere construction
from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from pathlib import Path
from pymoliere.construct import file_util
from pymoliere.ml import train_model, evaluate_model
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


if __name__ == "__main__":
  config = cpb.SentenceClassifierConfig()
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
  shared_scratch = Path(config.shared_scratch)
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
  # Need to make sure model_path is writable
  model_path.parent.mkdir(parents=True, exist_ok=True)

  # We're going to store model-specific checkpoints separately
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

  # Training data is ready, time to go!
  print("Prepping model")
  if torch.cuda.is_available() and not config.sys.disable_gpu:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  model = SentenceClassifier()
  # if torch.cuda.device_count() and not config.sys.disable_gpu:
    # print("Going to two gpu")
    # model = torch.nn.DataParallel(model)
  print("Model -> Device")
  model.to(device)

  loss_fn = torch.nn.NLLLoss()
  optimizer = torch.optim.Adam(
      filter(lambda x: x.requires_grad, model.parameters()),
      lr=0.002,
  )


  if config.force_retrain or not model_path.is_file():
    # Get your data!
    print("Loading training data")
    training_data = file_util.load_to_memory(
        data_ckpt_dir.joinpath("training_data")
    )
    validation_data = file_util.load_to_memory(
        data_ckpt_dir.joinpath("validation_data")
    )
    print("Beginning Training")
    train_model.train_model(
        training_data=[x.dense_data for x in training_data],
        training_labels=[x.label for x in training_data],
        validation_data=[x.dense_data for x in validation_data],
        validation_labels=[x.label for x in validation_data],
        model=model,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=config.sys.num_epochs,
        batch_size=config.sys.batch_size,
        compute_accuracy=True,
    )
    del training_data
    del validation_data
    print("Saving model")
    torch.save(model.state_dict(), model_path)
  else:
    print("Loading Model")
    model.load_state_dict(torch.load(model_path))

  print(f"Loading test data from {data_ckpt_dir}")
  test_data = file_util.load_to_memory(
      data_ckpt_dir.joinpath("testing_data")
  )

  print("Evaluation")
  evaluate_model.evaluate_multiclass_model(
      model=model,
      device=device,
      batch_size=config.sys.batch_size,
      data=[x.dense_data for x in test_data],
      labels=[x.label for x in test_data],
      class_names=IDX2LABEL,
  )


from pathlib import Path
from agatha.config import config_pb2 as cpb

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
  training_db_dir = model_root_dir.joinpath("training_db")
  ngram_freqs_path = model_root_dir.joinpath("ngram_frequencies.pkl")

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

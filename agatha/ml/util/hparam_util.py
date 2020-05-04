from copy import deepcopy
from pathlib import Path
from argparse import Namespace

def remove_paths_from_namespace(hparams:Namespace):
  """Removes variables from the namespace that ends in _db, _dir, or _path

  The model is going to include all hparams in the checkpoint. This is a
  problem for path variables that are needed during training, but are not
  wanted in the release of the model. For instance, during training we are
  going to need to tell the model about the embeddings and helper database
  locations, as well as where to save the model. These paths are machine
  specific. When we release the model, or even when we start to move files
  around, these paths will not be consistent.

  Args:
    hparams: The result of calling parse_args.

  Returns:
    A copy of hparams with no variables ending in _db, _dir, or _path. Also
    removes any variables of type Path.

  """

  hparams = deepcopy(hparams)
  attributes = list(hparams.__dict__.keys())
  for attr in attributes:
    if (
        isinstance(getattr(hparams, attr), Path)
        or (
          attr.endswith("_db")
          or attr.endswith("_path")
          or attr.endswith("_dir")
        )
    ):
      delattr(hparams, attr)
  return hparams

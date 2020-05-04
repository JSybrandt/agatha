from argparse import ArgumentParser, Namespace
from agatha.ml.hypothesis_predictor import hypothesis_predictor as hp
from pytorch_lightning import Trainer
from socket import gethostname

if __name__ == "__main__":
  parser = hp.HypothesisPredictor.add_argparse_args(ArgumentParser())
  args = parser.parse_args()

  if args.verbose:
    print("Started training on:", gethostname())
    print(args)

  trainer = Trainer.from_argparse_args(args)
  model = hp.HypothesisPredictor(args)
  model.prepare_for_training()
  trainer.fit(model)

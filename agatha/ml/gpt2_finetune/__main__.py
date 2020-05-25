from argparse import ArgumentParser, Namespace
from agatha.ml.gpt2_finetune import gpt2_finetune as gptf
from pytorch_lightning import Trainer
from socket import gethostname

if __name__ == "__main__":
  parser = gptf.Gpt2Finetune.add_argparse_args(ArgumentParser())
  args = parser.parse_args()

  if args.verbose:
    print("Started training on:", gethostname())
    print(args)

  trainer = Trainer.from_argparse_args(args)
  model = gptf.Gpt2Finetune(args)
  model.prepare_for_training()
  trainer.fit(model)

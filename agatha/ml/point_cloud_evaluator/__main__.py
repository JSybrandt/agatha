import pytorch_lightning as pl
from argparse import ArgumentParser
from agatha.ml.point_cloud_evaluator.point_cloud_evaluator import (
    PointCloudEvaluator,
)
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--train-fraction", type=float, default=1)
  parser.add_argument("--train-num-machines", type=int, default=1)
  parser.add_argument("--train-gradient-clip-val", type=float, default=1)
  parser.add_argument(
      "--model-ckpt-dir",
      default=Path("./point_cloud_eval_ckpt")
  )


  PointCloudEvaluator.configure_argument_parser(parser)

  args = parser.parse_args()

  print(args)

  trainer = pl.Trainer(
      gradient_clip_val=args.train_gradient_clip_val,
      default_save_path=args.model_ckpt_dir,
      gpus=-1 if args.distributed else 1,
      nb_gpu_nodes=args.train_num_machines if args.distributed else 1,
      distributed_backend='ddp' if args.distributed else None,
      train_percent_check=args.train_fraction,
      val_percent_check=args.train_fraction,
      track_grad_norm=2 if args.debug else -1,
      overfit_pct=0.01 if args.debug else 0,
      weights_summary='full',
  )
  model = PointCloudEvaluator(args)
  trainer.fit(model)

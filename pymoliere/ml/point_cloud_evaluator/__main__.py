import pytorch_lightning as pl
from pymoliere.ml.point_cloud_evaluator.point_cloud_evaluator import (
    PointCloudEvaluator
)

if __name__ == "__main__":
  parser = PointCloudEvaluator.configure_argument_parser()
  trainer = pl.Trainer(
      gpus=1,
      #distributed_backend='dp',
  )
  with PointCloudEvaluator(parser.parse_args()) as model:
    trainer.fit(model)

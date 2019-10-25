import plotille
import torch
from typing import List, Tuple, Any, Callable, Generator, Dict
from os import system
from tqdm import tqdm
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence

# We call this on epoch start, starts with epoch number
OnEpochStartFn = Callable[[int], None]
# Called after calculating loss on training batch.
# Input is the loss value.
AfterLossCalculationFn = Callable[[torch.Tensor], None]
# A generator is created per-epoch. We're going to call this #batches times.
# We're going to assume this batch generator could go on forever
# Outputs the kwargs for the model, and the tensor we're comparing against
BatchGenerator = Generator[Tuple[Dict[Any, Any], torch.Tensor], None, None]
# Params are yield, send, return

# Given predicted batch and actual batch, produce a value. These are averaged
# per-epoch and recorded in line plots.
MetricFn = Callable[[torch.Tensor, torch.Tensor], float]

def print_line_plots(line_plots:List[Tuple[str,List[float]]])->None:
  """
  Example Usage:
  print_line_plots([
    ("Training": [1,2,3]),
    ("Validation": [2,1]),
  ])
  """
  colors = [
      "bright_blue",
      "bright_magenta",
      "bright_white",
      "bright_red",
      "bright_green",
      "bright_yellow",
  ]
  assert len(line_plots) <= len(colors)

  fig = plotille.Figure()
  fig.height = 10
  fig.set_x_limits(min_=0)
  for idx, (name, data) in enumerate(line_plots):
    color = colors[idx]
    fig.plot(
        list(range(len(data))),
        data,
        label=name,
        lc=color,
    )
  print(fig.show(legend=True))

def get_device_from_model(model:torch.nn.Module)->torch.device:
  return next(model.parameters()).device

def train_model(
    model:torch.nn.Module,
    loss_fn:torch.nn.modules.loss._Loss,
    num_epochs:int,
    on_epoch_start:OnEpochStartFn,
    batch_generator:BatchGenerator,
    optimizer:torch.optim.Optimizer=None,
    after_loss_calculation:AfterLossCalculationFn=None,
    num_batches:int=None,
    validation_num_batches:int=None,
    validation_batch_generator:BatchGenerator=None,
    metrics:Tuple[str, MetricFn]=None,
    disable_pbar:bool=False,
    disable_plots:bool=False,
)->None:
  """
  A generic training harness for pytorch models.

  Inputs
    - on_epoch_start: We call this to begin each epoch. An epoch starts with
      all plots.
    - metrics: We calculate these per batch, and keep running averages.  Note
      that loss is automatically added.
    - num_batches: If specified, we're only going to generate this many batches
      per epoch.
    - validation_num_batches: Is specified, we're only going to generate this
      many batches during validation.
    - batch_generator: A generator that produces input, expected_output pairs.
      If this goes forever, please pair with num_batches to avoid infinite
      loop.
    - validation_batch_generator: Same as batch_generator, but for validation.

    - after_loss_calculation: Called after calculating loss on a single batch.
      Used for more complicated strategies. If not specified, must specify an optimizer so we can set to default.

    - optimizer: If not set, we assume any optimization is being done in the
      after_loss_calculation callback. If set, we replace the
      after_loss_calculation with a default.

  """
  if num_batches is not None:
    assert num_batches > 0

  if validation_num_batches is not None:
    assert validation_num_batches > 0

  if after_loss_calculation is not None:
    assert optimizer is None

  if optimizer is not None:
    assert after_loss_calculation is None
    def default_update(loss):
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    after_loss_calculation = default_update

  if metrics is None:
    metrics = []

  phases = ["train"]
  if validation_batch_generator is not None:
    phases.append("validate")
  # place loss as the first metric
  metrics = [("loss", loss_fn)] + metrics
  metric2phase2values = {
      metric_name: {phase: [] for phase in phases}
      for metric_name, _ in metrics
  }

  for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs}")
    if on_epoch_start is not None:
      on_epoch_start(epoch)
    if not disable_plots:
      system("clear")
      for metric, phase2valuses in metric2phase2values.items():
        print(metric)
        print_line_plots(list(phase2valuses.items()))

    for phase in phases:
      if phase == "train":
        model.train()
        gen = batch_generator
        num = num_batches
      else:
        model.eval()
        gen = validation_batch_generator
        num = validation_num_batches

      metric2running_sum = {metric: 0.0 for metric, _ in metrics}
      running_total = 0.0

      device = get_device_from_model(model)

      pbar = tqdm(gen(), total=num, disable=disable_pbar)
      for batch_idx, (in_kwargs, expected_output) in enumerate(pbar):

        predicted_output = model(**in_kwargs)

        for metric_name, metric_fn in metrics:
          metric_val = metric_fn(predicted_output, expected_output)
          if metric_name == "loss":
            after_loss_calculation(metric_val)
          if isinstance(metric_val, torch.Tensor):
            metric_val = metric_val.detach()
          metric2running_sum[metric_name] += metric_val

        running_total += 1

        metric_desc_str = " ".join([
          f"{name}:{metric2running_sum[name]/running_total:0.4f}"
          for name in metric2running_sum
        ])
        desc = f"{phase} -- {metric_desc_str}"
        if not disable_pbar:
          pbar.set_description(desc)
        else:
          print(batch_idx, desc)

        if num_batches is not None and batch_idx >= num_batches - 1:
          break

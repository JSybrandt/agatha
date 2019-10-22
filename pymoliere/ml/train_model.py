import plotille
import torch
from torch import nn
from typing import List
from os import system
from tqdm import tqdm
from sklearn.utils import shuffle
from pymoliere.util.misc_util import iter_to_batches
from torch.nn.utils.rnn import pad_sequence


def train_model(
    model:nn.Module,
    device:torch.device,
    loss_fn:nn.modules.loss._Loss,
    optimizer:torch.optim.Optimizer,
    num_epochs:int,
    batch_size:int,
    training_data:List[torch.Tensor],
    training_labels:List[torch.Tensor],
    validation_data:List[torch.Tensor]=None,
    validation_labels:List[torch.Tensor]=None,
    shuffle_batch:bool=True,
    compute_accuracy:bool=False,
    input_is_sequences:bool=False,
    show_plots:bool=True,
)->None:
  training_losses = []
  validation_losses = []
  training_accuracies = []
  validation_accuracies = []
  phases = ["train"]
  if validation_data is not None and validation_labels is not None:
    phases.append("validation")

  def print_training_plot(training_data, validation_data):
    fig = plotille.Figure()
    fig.height = 10
    fig.set_x_limits(min_=0)
    fig.plot(
        list(range(len(training_data))),
        training_data,
        label="Training",
        lc="bright_blue",
    )
    fig.plot(
        list(range(len(validation_data))),
        validation_data,
        label="Validation",
        lc="bright_magenta",
    )
    print(fig.show(legend=True))

  for epoch in range(num_epochs):
    if shuffle_batch:
      print("Shuffling...")
      training_data, training_labels = shuffle(training_data, training_labels)
    for phase in phases:
      if show_plots:
        system("clear")
        print(f"Epoch: {epoch}/{num_epochs} -- {phase}")
        print("Loss")
        print_training_plot(training_losses, validation_losses)
        if compute_accuracy:
          print("Accuracy")
          print_training_plot(training_accuracies, validation_accuracies)
        print()

      if phase == "train":
        model.train()
        losses = training_losses
        accuracies = training_accuracies
        X = training_data
        y_true = training_labels
      else:
        model.eval()
        losses = validation_losses
        accuracies = validation_accuracies
        X = validation_data
        y_true = validation_labels

      running_loss = 0.0
      running_corrects = 0.0
      running_count = 0.0

      def get_desc():
        if running_count == 0:
          return "-"*5
        else:
          l = running_loss/running_count
          desc = f"Loss:{l:0.4f} "
          if compute_accuracy:
            a = float(running_corrects)/running_count
            desc += f"Acc:{a:0.4f} "
          return desc
      pbar = tqdm(
          iter_to_batches(zip(X, y_true), batch_size),
          total=int(len(X)/batch_size)
      )
      for batch in pbar:
        vals = [x for x, _ in batch]
        if input_is_sequences:
          batch_data = pad_sequence(vals, batch_first=True)
        else:
          batch_data = torch.stack([x for x,_ in batch])

        batch_labels = torch.stack([y for _, y in batch])

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_data)
        loss = loss_fn(predictions, batch_labels)

        if phase == "train":
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        running_loss += loss.detach() * batch_data.size(0)
        running_count += float(batch_data.size(0))
        if compute_accuracy:
          _, batch_predictions = torch.max(predictions, 1)
          running_corrects += float(torch.sum(batch_predictions == batch_labels))
        pbar.set_description(get_desc())

      losses.append(float(running_loss / running_count))
      if compute_accuracy:
        accuracies.append(float(running_corrects / running_count))

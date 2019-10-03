import plotille
import torch
from torch import nn
from typing import List
from os import system
from tqdm import tqdm
from sklearn.utils import shuffle
from pymoliere.util.misc_util import iter_to_batches


def train_classifier(
    model:nn.Module,
    device:torch.device,
    loss_fn:nn.modules.loss._Loss,
    optimizer:torch.optim.Optimizer,
    num_epochs:int,
    training_data:List[torch.Tensor],
    validation_data:List[torch.Tensor],
    training_labels:List[int],
    validation_labels:List[int],
    batch_size:int,
    shuffle_batch:bool=True,
)->None:
  training_losses = []
  validation_losses = []
  training_accuracies = []
  validation_accuracies = []

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
    for phase in ["train", "validation"]:
      system("clear")
      print(f"Epoch: {epoch}/{num_epochs} -- {phase}")
      print("Loss")
      print_training_plot(training_losses, validation_losses)
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
          a = float(running_corrects)/running_count
          return f"Loss:{l:0.4f} Acc:{a:0.4f}"

      pbar = tqdm(
          iter_to_batches(zip(X, y_true), batch_size),
          total=int(len(X)/batch_size)
      )
      for batch in pbar:
        batch_data = torch.stack([x for x,_ in batch])
        batch_labels = torch.LongTensor([y for _, y in batch])

        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        batch_logits = model(batch_data)
        loss = loss_fn(batch_logits, batch_labels)

        if phase == "train":
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        _, batch_predictions = torch.max(batch_logits, 1)
        running_loss += loss.detach() * batch_data.size(0)
        running_corrects += float(torch.sum(batch_predictions == batch_labels))
        running_count += float(batch_data.size(0))
        pbar.set_description(get_desc())

      losses.append(float(running_loss / running_count))
      accuracies.append(float(running_corrects / running_count))

import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from pathlib import Path
from tqdm import tqdm
from pymoliere.util.misc_util import iter_to_batches
from typing import List, Optional, Callable, Any, Dict

ToTensorFn = Callable[[Any], torch.Tensor]

def evaluate_multiclass_model(
    model:nn.Module,
    device:torch.device,
    batch_size:int,
    data:List[Any],
    labels:List[int],
    metadata:List[Any],
    class_names:List[str],
    output_dir:Path,
    data_batch_to_tensor_fn:Optional[ToTensorFn]=None,
)->None:
  """
  Given a pytorch model and some testing data, perform a number of multiclass
  evaluations. We populate the result dir with different evaluation summaries.
  @param metadata: Additional data that helps us understand each point beyond its simple vector information. Used for winners/losers
  @param class_names: a mapping of index2string
  """
  output_dir.mkdir(parents=True, exist_ok=True)

  model.to(device)
  model.eval()
  predicted_labels = []

  print("Generating Predictions")
  pbar = tqdm(
      iter_to_batches(data, batch_size),
      total=int(len(data)/batch_size)
  )
  for batch_data in pbar:
    if data_batch_to_tensor_fn is not None:
      batch_data = data_batch_to_tensor_fn(batch_data)
    batch_data = batch_data.to(device)

    batch_logits = model(batch_data)
    _, batch_predictions = torch.max(batch_logits, 1)
    batch_predictions = batch_predictions.detach().cpu().numpy().tolist()
    predicted_labels += batch_predictions

  confusion = confusion_matrix(labels, predicted_labels)
  p, r, f, s = precision_recall_fscore_support(labels, predicted_labels)
  for idx, name in enumerate(class_names):
    print(name)
    print(f"  - Precision: {p[idx]:0.5f}")
    print(f"  - Recall:    {r[idx]:0.5f}")
    print(f"  - F1:        {f[idx]:0.5f}")
    print(f"  - Support:   {s[idx]}")
    mistakes = [
        (n, c)
        for n, c
        in zip(class_names, confusion[idx, :])
        if n != name
    ]
    mistakes.sort(key=lambda x: x[1], reverse=True)
    print("  - Mistakes:")
    for n, c in mistakes:
      print(f"    - {n} {c}")

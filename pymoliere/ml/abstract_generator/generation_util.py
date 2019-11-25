from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer,
    AbstractGenerator,
)
from pymoliere.util.misc_util import Record, iter_to_batches
import torch
from typing import Tuple, List, Iterable
try:
  from nlgeval import NLGEval
except ImportError:
 # This is a heavy dependency, and we don't want to worry all users with it.
 pass

def generate_new_text(
  model:AbstractGenerator,
  tokenizer:AbstractGeneratorTokenizer,
  context:torch.LongTensor,
  text:torch.LongTensor=None,
)->Iterable[Tuple[str, str]]:
  # Can give both or neither
  if text is None:
    text = (
        torch.LongTensor([tokenizer.start_symbol_idx])
        .unsqueeze(1)
        .to(context.device)
    )

  # Only supporting batch size 1 for now
  assert context.shape[1] == text.shape[1] == 1

  while True:
    predictions = model(context, text)

    new_word = predictions["text"][-1, 0, :].argmax() \
        + tokenizer.vocab_start_idx

    yield int(new_word)

    def add_or_shift(tensor, new_element):
      l = tensor.flatten().tolist()
      l.append(new_element)
      if len(l) >= model.max_text_length:
        l = l[-model.max_text_length+1:]
      return torch.LongTensor(l).unsqueeze(1).to(tensor.device)

    text = add_or_shift(text, int(new_word))

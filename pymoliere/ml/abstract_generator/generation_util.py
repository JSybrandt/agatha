from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer,
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
  model:torch.nn.Module,
  tokenizer:AbstractGeneratorTokenizer,
  context:torch.LongTensor,
  text:torch.LongTensor,
  types:torch.LongTensor,
)->Iterable[Tuple[str, str]]:

  # Only supporting batch size 1 for now
  assert text.shape[0] == types.shape[0]
  assert context.shape[1] == text.shape[1] == types.shape[1] == 1

  first_run = True
  while True:
    predictions = model(context, text, types)

    if first_run:
      first_run = False
      inner_words = predictions["text"][:-1, 0, :].argmax(dim=1) \
          + tokenizer.vocab_start_idx
      inner_types = predictions["types"][:-1, 0, :].argmax(dim=1) \
          + tokenizer.sent_type_start_idx
      for idx in range(len(inner_words)):
        yield(int(inner_words[idx]), int(inner_types[idx]))

    new_word = predictions["text"][-1, 0, :].argmax() \
        + tokenizer.vocab_start_idx

    new_type = predictions["types"][-1, 0, :].argmax() \
        + tokenizer.sent_type_start_idx

    yield (
        int(new_word),
        int(new_type)
    )

    tmp_text = text.clone()
    text[:-1] = tmp_text[1:]
    text[-1, 0] = new_word

    tmp_types = types.clone()
    types[:-1] = tmp_types[1:]
    types[-1, 0] = new_type



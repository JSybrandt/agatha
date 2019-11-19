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

  tmp_text = text.clone()
  tmp_types = types.clone()

  while True:
    text[:-1, 0] = tmp_text[1:, 0]
    text[-1, 0] = tokenizer.mask_idx
    types[:-1, 0] = tmp_types[1:, 0]
    types[-1, 0] = tokenizer.mask_idx

    predictions = model(context, text, types)

    new_word = predictions["text"][-1, 0, :].argmax() \
        + tokenizer.vocab_start_idx

    new_type = predictions["types"][-1, 0, :].argmax() \
        + tokenizer.sent_type_start_idx

    yield (
        int(new_word),
        int(new_type)
    )

    tmp_text = text.clone()
    tmp_text[-1, 0] = new_word

    tmp_types = types.clone()
    tmp_types[-1, 0] = new_type



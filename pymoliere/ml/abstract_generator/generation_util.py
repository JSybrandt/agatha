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

  while True:
    text = torch.LongTensor(text.flatten().tolist() + [tokenizer.mask_idx]).unsqueeze(1)
    types = torch.LongTensor(types.flatten().tolist() + [tokenizer.mask_idx]).unsqueeze(1)
    if text.shape[0] > model.max_text_length:
      text = text[-model.max_text_length:]
      types = types[-model.max_text_length:]
    predictions = model(context, text, types)
    new_word = predictions["text"][-1, 0].argmax() \
        + tokenizer.vocab_start_idx
    new_type = predictions["types"][-1, 0].argmax() \
        + tokenizer.sent_type_start_idx
    print(new_word, new_type)
    yield (
        tokenizer.decode_idx(int(new_word)),
        tokenizer.decode_idx(int(new_type))
    )
    text[-1, 0] = new_word
    types[-1, 0] = new_type



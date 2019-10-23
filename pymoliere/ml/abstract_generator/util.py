from typing import Dict, Any, Iterable
from pymoliere.util.misc_util import Record
from transformers import BertModel, BertTokenizer
import torch
from copy import copy
from typing import List
from random import random

class AbstractGenerator(BertModel):
  def __init__(self, config:Dict[str, Any]):
    super(AbstractGenerator, self).__init__(config)
    for param in self.parameters():
      param.requires_grad = False
    # This last layer converts the hidden layer to a predicted word
    self.last_hidden2voccab = torch.nn.Linear(
        config.hidden_size,
        config.vocab_size,
    )
    # Then we pick the word
    self.last_softmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, x):
    # x is batch_first
    batch_size, seq_len = x.shape
    # The 0th element is the hidden layer per-word
    x = super(AbstractGenerator, self).forward(x)[0]

    # Convert to (batch_size*seq_len), hidden_size to apply linear layer to
    # each row
    x = x.view((batch_size*seq_len), self.config.hidden_size)
    # apply prediction
    x = self.last_hidden2voccab(x)
    x = self.last_softmax(x)
    # reconstitute correct shape
    x = x.view(batch_size, seq_len, self.config.vocab_size)
    return x

def group_sentences_into_pairs(records:Iterable[Record])->Iterable[Record]:
  """
  Expect records to have the following fields:
        "sent_text": str,
        "sent_idx": int,
        "sent_total": int,
        "pmid": int,
        "version": int,
  """
  res = []
  key2text = {}
  for record in records:
    key = (record["pmid"], record["version"], record["sent_idx"])
    value = record["sent_text"]
    # Don't want dups
    assert key not in key2text
    key2text[key] = value
    prev_key = (record["pmid"], record["version"], record["sent_idx"]-1)
    next_key = (record["pmid"], record["version"], record["sent_idx"]+1)
    if prev_key in key2text:
      res.append((key2text[prev_key], key2text[key]))
    if next_key in key2text:
      res.append((key2text[key], key2text[next_key]))
  return res

def mask_sequence(
    tokenizer:BertTokenizer,
    sequence:List[int],
    mask:List[bool],
)->List[int]:
  """
  If mask[i] is True, then we replace sequence[i] with tokenizer.mask_token_id.
  Returns a COPY
  """

  sequence = copy(sequence)
  assert len(sequence) == len(mask)
  for idx in range(len(sequence)):
    if mask[idx]:
      sequence[idx] = tokenizer.mask_token_id
  return sequence

def generate_sentence_mask(
    tokenizer:BertTokenizer,
    sequence:List[int],
    per_token_mask_prob:float,
)->List[bool]:
  assert per_token_mask_prob >= 0
  assert per_token_mask_prob <= 1
  assert sequence[0] == tokenizer.cls_token_id
  assert sequence[-1] == tokenizer.sep_token_id
  mid_sep_idx = sequence.find(tokenizer.sep_token_id)
  # Must find a separation token between the start and end
  assert mid_sep_idx < len(sequence)-1
  # init mask to 0
  mask = [False] * len(sequence)
  for idx in range(mid_sep_idx+1, len(sequence)-1):
    if random() <= per_token_mask_prob:
      mask[idx] = True
  return mask

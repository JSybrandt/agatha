from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
from transformers import BertModel, BertTokenizer
import torch
from copy import copy
from typing import List
import numpy as np
from pymoliere.ml.train_model import get_device_from_model
from pymoliere.util.misc_util import flatten_list

class AbstractGenerator(BertModel):
  def __init__(self, config:Dict[str, Any], freeze_bert_layers=False):
    super(AbstractGenerator, self).__init__(config)
    for param in self.parameters():
      param.requires_grad = not freeze_bert_layers
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
  mid_sep_idx = sequence.index(tokenizer.sep_token_id)
  # Must find a separation token between the start and end
  assert mid_sep_idx < len(sequence)-1
  # init mask to 0
  mask = [False] * len(sequence)
  for idx in range(mid_sep_idx+1, len(sequence)-1):
    if np.random.random() <= per_token_mask_prob:
      mask[idx] = True
  return mask

def sentence_pairs_to_tensor_batch(
    tokenizer:BertTokenizer,
    batch_pairs:List[Tuple[str, str]],
    unchanged_prob:float,
    full_mask_prob:float,
    mask_per_token_prob:float,
    max_sequence_length:int,
)->Tuple[torch.Tensor, torch.Tensor]:
  """
  Converts the sentence pairs into training batches.
  Input:
    tokenizer: the object that we will use to convert raw text into sequences
    sentence_pairs: the raw textual data. The second sentence of each pair is
    the one we would like to predict.
    unchanged_prob: number from 0 - 1. This is the chance that we do not mask
    the second sentence at all. unchanged_prob + full_mask_prob < 1
    full_mask_prob: number from 0 - 1. This is the chance that we mask the
    entire sentence. unchanged_prob + full_mask_prob < 1.
    mask_per_token_prob: number from 0 - 1. This is the chance that we mask
    each word, provided this sentence isn't ignored or totally masked.
  Output:
    (modified_data, original_data)
    Model input is a batch of padded sequences wherein the second sentence may
    have been modified.  Expected output is the original padded sequences with
    no modification.
  """
  assert unchanged_prob + full_mask_prob <= 1

  def pick_mask_prob():
    r = np.random.random()
    if r < unchanged_prob:
      return 0
    r -= unchanged_prob
    if r < full_mask_prob:
      return 1
    return mask_per_token_prob

  original_sequences = [
      tokenizer.encode(
        text=first,
        text_pair=second,
        add_special_tokens=True,
        max_length=max_sequence_length
      )
      for first, second in batch_pairs
  ]

  masked_sequences = [
      mask_sequence(
        tokenizer=tokenizer,
        sequence=sequence,
        mask=generate_sentence_mask(
          tokenizer=tokenizer,
          sequence=sequence,
          per_token_mask_prob=pick_mask_prob()
        )
      )
      for sequence in original_sequences
  ]

  original_sequences = list(map(torch.tensor, original_sequences))
  masked_sequences = list(map(torch.tensor, masked_sequences))

  modified_data = torch.nn.utils.rnn.pad_sequence(
      sequences=masked_sequences,
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  original_data = torch.nn.utils.rnn.pad_sequence(
      sequences=original_sequences,
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  return modified_data, original_data


def generate_sentence(
    sentence:str,
    model:AbstractGenerator,
    tokenizer:BertTokenizer,
    max_sequence_length:int,
)->str:
  device = get_device_from_model(model)
  # Sequence holds the tokens for both input and mask
  sequence, _ = sentence_pairs_to_tensor_batch(
      tokenizer=tokenizer,
      batch_pairs=[(sentence, sentence)],
      unchanged_prob=0,
      full_mask_prob=1,
      mask_per_token_prob=1,
      max_sequence_length=max_sequence_length,
  )
  sequence = sequence.to(device)

  # Get where the separation between sequences happens
  sep_indices = (sequence == tokenizer.sep_token_id).nonzero().tolist()
  first_predicted_idx = 1 + min([c for r, c in sep_indices])

  # False for each element in the 2nd half of the sequence
  complete_mask = [False] * (sequence.shape[1]-1-first_predicted_idx)

  while False in complete_mask:
    # Predict based off what we have currently
    # make a list of softmax results (seq_len x voccab_size)
    predicted = model(sequence).view(-1, tokenizer.vocab_size)[first_predicted_idx:-1]
    # how confident were we at each token?
    confidence_per_token, token_indices = torch.max(predicted, dim=1)
    # If we've already settled on a word, ignore it
    confidence_per_token[complete_mask] = -float("inf")
    # Which word are we the most confident about?
    selected_idx = torch.argmax(confidence_per_token)
    complete_mask[selected_idx] = True
    # Remember this index!
    sequence[0, selected_idx + first_predicted_idx] = token_indices[selected_idx]

  return tokenizer.decode(sequence[0, first_predicted_idx:-1].tolist())





from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
import torch
from copy import copy
from typing import List
from pymoliere.ml.train_model import get_device_from_model
from pymoliere.util.misc_util import flatten_list
from random import random, randint, shuffle
from datetime import datetime
import pickle
from pymoliere.ml.abstract_generator.abstract_generator_util import (
    AbstractGeneratorTokenizer
)
from itertools import accumulate

try:
  from nlgeval import NLGEval
except ImportError:
  # This is a heavy dependency, and we don't want to worry all users with it.
  pass




def abstract_to_training_data(
    abstract:Record,
    tokenizer:AbstractGeneratorTokenizer,
    seed_text_size:int,
    follow_text_size:int,
)->Tuple[List[int], List[int]]:
  """
  Given an abstract, randomly select a seed and target.
  Process these text windows using the tokenizer.
  Return seed and target token sequences
  """

  year = int(abstract["date"].split("-")[0])
  mesh_headings = abstract["mesh_headings"]
  authors = abstract["authors"]

  # Tokenize the whole thing
  tokens_per_field = [
      tokenizer.encode_text(t["text"])
      for t in abstract["text_data"]
  ]

  # The abstract-level index of sentence i begins at prog_sizes[i] and ends at 
  # prog_sizes[i-1]
  progressive_sizes = [0] + list(accumulate(map(len, tokens_per_field)))

  # Begin selecting somewhere in the sentence
  selection_start_idx = random.randint(0, progressive_sizes[-1])
  # finish selecting at the end
  selection_end_idx = min(
      progressive_sizes[-1],
      selection_start_idx + seed_text_size + follow_text_size
  )
  # Want to keep the seed - follow ratio in the case that we've got too small a
  # selection
  seed_to_total_ratio = seed_text_size / (seed_text_size + follow_text_size)
  selection_middle_idx = int(
      (selection_end_idx - selection_start_idx)
      * seed_to_total_ratio
      + selection_start_idx
  )

  selection_start_idx = \
    progressive_sizes[-1] - seed_text_size - follow_text_size
  if selection_start_idx < 0:
    selection_start_idx = 0



  ###################


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

def apply_mask_to_token_ids(
    tokenizer:BertTokenizer,
    input_ids:List[int],
    mask:List[bool],
)->List[int]:
  """
  If mask[i] is True, then we replace input_ids[i] with tokenizer.mask_token_id.
  Returns a COPY
  """
  input_ids = copy(input_ids)
  assert len(input_ids) == len(mask)
  for idx in range(len(input_ids)):
    if mask[idx]:
      input_ids[idx] = tokenizer.mask_token_id
  return input_ids


def apply_random_replace_to_token_ids(
    tokenizer:BertTokenizer,
    input_ids:List[int],
    mask:List[bool],
)->List[int]:
  """
  If mask[i] is true, we replace input_ids[i] with a random token. Returns a
  copy.
  """
  def is_bad_token(token:str):
    # This check makes it so we don't get [PAD] or [unused...]
    return len(token)==0 or (token[0] == "[" and token[-1] == "]")

  def get_random_token_id():
    t = randint(0, tokenizer.vocab_size-1)
    while is_bad_token(tokenizer.convert_ids_to_tokens(t)):
      t = randint(0, tokenizer.vocab_size-1)
    return t

  input_ids = copy(input_ids)
  assert len(input_ids) == len(mask)
  for idx in range(len(input_ids)):
    if mask[idx]:
      input_ids[idx] = get_random_token_id()
  return input_ids


def sentence_pairs_to_model_io(
    tokenizer:BertTokenizer,
    batch_pairs:List[Tuple[str, str]],
    unchanged_prob:float,
    full_mask_prob:float,
    mask_per_token_prob:float,
    replace_per_token_prob:float,
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
    replace_per_token_prob: The rate that we randomly alter a word.
    Note that mask_per_token_prob + replace_per_token_prob < 1
  Output:
    (model_kwargs, original_data)
    Model input is a batch of padded sequences wherein the second sentence may
    have been modified.  Expected output is the original padded sequences with
    no modification.

    model kwargs contains:
      input_ids: the set of tokens for two sentences. Starts with [CLS] and [SEP] denotes the end as well as the intermediate sentence.
      attention_mask: all 1's until padding tokens. Avoids running attention on padding.
      token_type_ids: all 0's until we reach the 2nd sentence (starting after the middle [SEP]) and then all 1's.
  """
  assert unchanged_prob + full_mask_prob <= 1
  assert replace_per_token_prob + mask_per_token_prob <= 1

  # When we use this value, we're going to be setting the replacement
  # Given that we've not masked the token
  if mask_per_token_prob == 1:
    replace_per_token_prob = 0
  else:
    replace_per_token_prob /= (1-mask_per_token_prob)

  def pick_mask_prob():
    r = random()
    if r < unchanged_prob:
      return 0.0
    r -= unchanged_prob
    if r < full_mask_prob:
      return 1.0
    return mask_per_token_prob

  # Original inputs contains:
  # 'input_ids' 'token_type_ids' and 'special_tokens_mask'
  model_inputs = [
      tokenizer.encode_plus(
        text=first,
        text_pair=second,
        add_special_tokens=True,
        max_length=max_sequence_length
      )
      for first, second in batch_pairs
  ]

  modified_input_ids = []
  for kwargs in model_inputs:
    # 1 if we're allowed to mask here
    mask_positions = kwargs["token_type_ids"]
    # Can't mask last sep
    mask_positions[-1] = 0
    mask_positions= torch.FloatTensor(mask_positions)

    # The mask is a bernoulli sample over valid positions
    mask_mask = torch.bernoulli(mask_positions * pick_mask_prob())

    # Now we need the mask of randomly replaced tokens
    # These cannot be those positions that have already been masked
    mask_positions = mask_positions * (1-mask_mask)
    replace_mask = torch.bernoulli(mask_positions * replace_per_token_prob)

    # Need to make sure that the mask's are disjoint
    assert (mask_mask * replace_mask).sum() == 0

    ids = kwargs["input_ids"]
    ids = apply_mask_to_token_ids(
        tokenizer=tokenizer,
        input_ids=ids,
        mask=mask_mask.bool().tolist(),
    )
    ids = apply_random_replace_to_token_ids(
        tokenizer=tokenizer,
        input_ids=ids,
        mask=replace_mask.bool().tolist(),
    )
    modified_input_ids.append(ids)

  original_token_ids = [
      torch.tensor(arg['input_ids'], dtype=int)
      for arg in model_inputs
  ]
  token_type_ids = [
      torch.tensor(arg['token_type_ids'])
      for arg in model_inputs
  ]
  modified_input_ids = [torch.LongTensor(ids) for ids in modified_input_ids]

  def pad(x):
    return torch.nn.utils.rnn.pad_sequence(
        sequences=x,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
  original_token_ids = pad(original_token_ids)
  token_type_ids = pad(token_type_ids)
  modified_input_ids = pad(modified_input_ids)
  attention_mask = torch.ones(original_token_ids.shape)
  attention_mask[original_token_ids == tokenizer.pad_token_id] = 1

  return (
      {
        "input_ids": modified_input_ids,
        "token_type_ids": token_type_ids,
        'attention_mask': attention_mask,
      },
      original_token_ids)
  return (modified_data, token_types), original_data


def generate_sentence(
    sentence:str,
    model:AbstractGenerator,
    tokenizer:BertTokenizer,
    max_sequence_length:int,
    generated_sentence_length:int=None,
    reference_result_sentence:str=None,
)->str:
  device = get_device_from_model(model)
  # Sequence holds the tokens for both input and mask
  generated_template = sentence
  if generated_sentence_length is not None:
    generated_template = "x "*generated_sentence_length
  if reference_result_sentence is not None:
    generated_template = reference_result_sentence
  model_kwargs, _ = sentence_pairs_to_model_io(
      tokenizer=tokenizer,
      batch_pairs=[(sentence, generated_template)],
      unchanged_prob=0,
      full_mask_prob=1,
      replace_per_token_prob=0,
      mask_per_token_prob=1,
      max_sequence_length=max_sequence_length,
  )
  # Send everything to appropriate device
  model_kwargs = {k: v.to(device) for k, v in model_kwargs.items()}

  first_predicted_idx = (
      model_kwargs["token_type_ids"]
      .view(-1)
      .tolist()
      .index(1)
  )
  complete_mask = (
      [False]
      * (model_kwargs["input_ids"].shape[1] - first_predicted_idx - 1)
  )

  while False in complete_mask:
    # Predict based off what we have currently
    # make a list of softmax results (seq_len x voccab_size)
    predicted = (
        model(**model_kwargs)
        .view(-1, tokenizer.vocab_size)
        [first_predicted_idx:-1]
    )
    # how confident were we at each token?
    confidence_per_token, token_indices = torch.max(predicted, dim=1)
    # If we've already settled on a word, ignore it
    confidence_per_token[complete_mask] = -float("inf")
    # Which word are we the most confident about?
    selected_idx = torch.argmax(confidence_per_token)
    complete_mask[selected_idx] = True
    # Remember this index!
    model_kwargs['input_ids'][0, selected_idx+first_predicted_idx] =\
        token_indices[selected_idx]

  return tokenizer.decode(
      model_kwargs['input_ids'][0, first_predicted_idx:-1].tolist()
  )


def evaluate_generation(
  initial_sentence:str,
  follow_sentence:str,
  generated_sentence:str,
)->Dict[str,float]:
  if not hasattr(evaluate_generation, "nlgeval"):
    print("Initializing eval models")
    evaluate_generation.nlgeval = NLGEval()
  return  evaluate_generation.nlgeval.compute_individual_metrics(
      [follow_sentence],
      generated_sentence,
  )




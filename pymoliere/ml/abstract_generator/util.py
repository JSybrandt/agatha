from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
import torch
from copy import copy
from typing import List
from pymoliere.ml.train_model import get_device_from_model
from pymoliere.util.misc_util import flatten_list
from random import random, randint, shuffle
from datetime import datetime
import sentencepiece as spm
import pickle

try:
  from nlgeval import NLGEval
except ImportError:
  # This is a heavy dependency, and we don't want to worry all users with it.
  pass

MODEL_NAME = "abstract_generator"

INTERESTING_SENTENCE_LABLES = {
    "title": 0,
    "abstract:background": 1,
    "abstract:conclusions": 2,
    "abstract:methods": 3,
    "abstract:objective": 4,
    "abstract:results": 5,
}
INDEX_TO_SENTENCE_LABEL = [
    "title",
    "abstract:background",
    "abstract:conclusions",
    "abstract:methods",
    "abstract:objective",
    "abstract:results",
]

class AbstractGeneratorTokenizer(object):
  def __init__(self, tokenizer_model_path:Path, extra_data_path:Path):
    self.sp_processor = spm.SentencePieceProcessor()
    if not self.sp_processor.load(tokenizer_model_path):
      raise ValueError("Invalid model path", tokenizer_model_path)
    with open(extra_data_path, "rb") as f:
      extra_data = pickle.load(f)
    # the idx_to_* works because these are ordered dicts
    self.author_index = extra_data["author_index"]
    self.idx_to_author = list(author_index)
    self.mesh_index = extra_data["mesh_index"]
    self.idx_to_mesh = list(mesh_index)
    self.oldest_year = extra_data["oldest_year"]

    self.author_size = len(self.author_index)
    self.vocab_size = len(self.sp_processor)
    self.mesh_size = len(self.mesh_index)

    self.padding_idx = 0
    self.unknown_idx = 1
    self.start_idx = 2
    self.sep_idx = 3
    self.mask_idx = 4
    self.secial_markers = ["[PAD]", "[UNK]", "[START]", "[SEP]", "[MASK]"]
    self.special_size = 5
    self.special_start_idx = 0
    self.special_end_idx = self.special_start_idx + self.special_size

    # Sentence Types
    self.sent_type_size = len(INTERESTING_SENTENCE_LABLES)
    self.sent_type_start_idx = self.special_end_idx
    self.sent_type_end_idx = self.sent_type_size
    # Dates
    self.date_size = datetime.now().year - self.oldest_year
    self.date_start_idx = self.sent_type_end_idx
    self.date_end_idx = self.date_start_idx + self.date_size
    # Authors
    self.author_size = author_size
    self.author_start_idx = self.date_end_idx
    self.author_end_idx = self.author_start_idx + self.author_size
    # Mesh terms
    self.mesh_size = mesh_size
    self.mesh_start_idx = self.author_end_idx
    self.mesh_end_idx = self.mesh_start_idx + self.mesh_size
    # Voccab
    self.vocab_size = vocab_size
    self.vocab_start_idx = self.mesh_end_idx
    self.vocab_end_idx = self.vocab_start_idx + self.vocab_size

    self.total_index_size = self.vocab_end_idx

  def encode_author(self, author_name:str)->int:
    if author_name is None:
      return self.padding_idx
    if author_name in self.author_index:
      return self.author_index[author_name] + self.author_start_idx
    else:
      return self.unknown_idx

  def encode_mesh(self, mesh_code:str)->int:
    if mesh_code is None:
      return self.padding_idx
    if mesh_code in self.mesh_index:
      return self.mesh_index[mesh_code] + self.mesh_start_idx
    else:
      return self.unknown_idx

  def encode_year(self, year:Optional[int])->int:
    if year is None:
      return self.padding_idx
    if year < self.oldest_year or year > datetime.now().year:
      return self.unknown_idx
    return year - self.oldest_year + self.data_start_idx

  def encode_text(self, text:str)->List[int]:
    if text is None:
      return []
    return [
        token + self.vocab_start_idx
        for token in self.sp_processor.encode_as_ids(text)
    ]

  def encode_sent_type(self, sent_type:str)->int:
    if sent_type is None:
      return self.padding_idx
    if sent_type not in INTERESTING_SENTENCE_LABLES:
      return self.unknown_idx
    return INTERESTING_SENTENCE_LABLES[sent_type] + self.sent_type_start_idx

  def decode_idx(self, index:int)->str:
    if 0 <= idx < self.special_end_idx:
      return self.special_markers[idx]
    if self.sent_type_start_idx <= idx < self.sent_type_end_idx:
      return INDEX_TO_SENTENCE_LABEL[idx - self.sent_type_start_idx]
    if self.date_start_idx <= idx < self.date_end_idx:
      return str(idx - self.date_start_idx + self.oldest_year)
    if self.author_start_idx <= idx < self.author_end_idx:
      return self.idx_to_author[idx - self.author_start_idx]
    if self.mesh_start_idx <= idx < self.mesh_end_idx:
      return self.idx_to_meshy[idx - self.mesh_start_idx]
    if self.vocab_start_idx <= idx < self.vocab_end_idx:
      return self.sp_processor.id_to_piece(idx - self.vocab_start_idx)
    return "[INVALID]"

  def decode_text(self, indices:List[int])->str:
    return self.sp_processor.decode_ids([
      idx - self.vocab_start_idx
      for idx in indices
      if self.vocab_start_idx <= idx < self.vocab_end_idx
    ])

  def encode_all(
      self,
      required_author_count:int,
      required_mesh_count:int,
      max_text_length:int,
      date:int=None,
      start_sentence_type:str=None,
      end_sentence_type:str=None,
      authors:List[str]=None,
      mesh_headings:List[str]=None,
      text:str=None,
  )->List[int]:
    date_idx = self.encode_date(date)
    start_type_idx = self.encode_sent_type(start_sentence_type)
    end_type_idx = self.encode_sent_type(end_sentence_type)
    author_indices = [self.encode_author(a) for a in authors]
    mesh_indices = [self.encode_mesh(m) for m in mesh_headings]
    text_indices = self.encode_text(text)

    # Subset text if nessesary
    text_indices = text_indices[:max_text_length]

    def to_required_size(data, size):
      # pad if nessesary
      while(len(data) < size):
        data.append(self.padding_idx)
      # randomize
      shuffle(data)
      # If too long, remove extra
      del data[size:]

    to_required_size(author_indices, required_author_count)
    to_required_size(mesh_indices, required_mesh_count)

    return (
        [
          self.start_idx
          date_idx,
          start_type_idx,
          end_type_idx,
        ]
        + author_indices
        + mesh_indices
        + [self.sep_idx]
        + text_indices
        + [self.sep_idx]
    )



  def decode_all(
      self,
      indices:List[int],
      required_author_count:int,
      required_mesh_count:int,
  )->Dict[str, Any]:
    # skip start at 0
    date_idx = indices[1]
    start_type_idx = indices[2]
    end_type_idx = indices[3]
    indices = indices[4:]
    # slice out authors
    author_indices = indices[:required_author_count]
    indices = indices[required_author_count:]
    # slice out mesh
    mesh_indices = indices[:required_mesh_count]
    indices = indices[required_mesh_count:]
    # skip sep at 0 and -1
    text_indices = indices[1:-1]

    # ignore padding
    author_indices = [a for a in author_indices if a != self.padding_idx]
    mesh_indices = [a for a in mesh_indices if a != self.padding_idx]
    text_indices = [a for a in text_in dices if a != self.padding_idx]

    return {
        "date": self.decode_idx(date_idx),
        "start_sent_type": self.decode_idx(start_type_idx),
        "end_sent_type": self.decode_idx(end_type_idx),
        "authors": [self.decode_idx(x) for x in author_indices],
        "mesh_headings": [self.decode_idx(x) for x in mesh_headings],
        "text": self.decode_text(text_indices),
    }


class AbstractGenerator(torch.nn.Module):
  def __init__(self,
      embedding_size:int,
      embedding_dim:int,
      num_attention_heads:int,
      num_encoder_layers:int,
      num_decoder_layers:int,
      intermediate_dropout:float,
      intermediate_feedforward_dim:int,
      max_mesh_terms:int,
      max_authors:int,
  ):
    """
    Learns to generate following text given sliding windows across abstracts.
    """
    super(AbstractGenerator, self).__init__()

    self.embeddings = torch.nn.Embedding(
        embedding_size,
        embedding_dim,
        padding_index=0,
        max_norm=1,
    )

    self.transformer = torch.nn.Transformer(
        d_model=embedding_dim,
        nhead=num_attention_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=intermediate_feedforward_dim,
        dropout=intermediate_dropout,
    )

    self.predict_output = torch.nn.Linear(
        embedding_dim,
        embedding_size,
    )

    self.softmax = torch.nn.LogSoftmax(dim=2)

  def forward(
      self,
      seed:torch.LongTensor,
      follow:torch.LongTensor,
  ):
    # S is the sequence length of seed
    # F is the sequence length of follow
    # B is the batch size
    # Expected shapes:
    #   - seed : (S, B)
    #   - follow : (F, B)

    # Set padding masks to ignore out-of-bound tokens
    # Relies on tokenizer using 0 as padding
    # padding is shape (B, S) and (B, F)
    seed_padding_mask = torch.zeros_like(seed)
    seed_padding_mask[seed == 0] = torch.float('-inf')
    seed_padding_mask.t_()  # in place transpose
    follow_padding_mask = torch.zeros_like(follow)
    follow_padding_mask[follow == 0] = torch.float('-inf').transpose()
    followin_padding_mask.t_()

    # E is the embedding dimensionality
    seed = self.embeddings(seed)
    # seed is now (S, B, E)
    follow = self.embeddings(follow)
    # follow is now (F, B, E)

    follow = self.transformer(
        src=seed,
        tgt=follow,
        src_key_padding_mask=seed_padding_mask,
        tgt_key_padding_mask=follow_padding_mask,
    )
    # Follow is still (F, B, E), however the transformer has been computed

    # V is the size of the "vocab" (total embedding size)
    follow = self.predict_output(follow)
    # follow is now (F, B, V)

    # produce softmax results across "vocab"
    return self.softmax(follow)


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




import torch
import pickle
from pymoliere.ml.abstract_generator.misc_util import (
    INTERESTING_SENTENCE_LABLES,
    INDEX_TO_SENTENCE_LABEL,
)
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
import math

def get_current_year():
  return datetime.now().year

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
    self.year_size = get_current_year() - self.oldest_year
    self.year_start_idx = self.sent_type_end_idx
    self.year_end_idx = self.year_start_idx + self.year_size
    # Authors
    self.author_size = author_size
    self.author_start_idx = self.year_end_idx
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
    if year < self.oldest_year or get_current_year():
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
    if self.year_start_idx <= idx < self.year_end_idx:
      return str(idx - self.year_start_idx + self.oldest_year)
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
      year:int=None,
      start_sentence_type:str=None,
      end_sentence_type:str=None,
      authors:List[str]=None,
      mesh_headings:List[str]=None,
      text:str=None,
  )->List[int]:
    year_idx = self.encode_year(year)
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
          year_idx,
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
    year_idx = indices[1]
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
        "year": self.decode_idx(year_idx),
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
      max_sequence_length:int,
      num_attention_heads:int,
      num_encoder_layers:int,
      num_decoder_layers:int,
      intermediate_dropout:float,
      intermediate_feedforward_dim:int,
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

    # Positional encoding is (Max Sequence Length, 1, Embedding Dim)
    self.positional_encoding = self.generate_positional_encoding(
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
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

  def generate_positional_encoding(
      self,
      max_sequence_length:int,
      embedding_dim:int,
  )->torch.FloatTensor:
    # Dim must be even
    assert embedding_dim % 0 == 0

    # Returns a (seq_len, emb) tensor
    positional_encodings = []
    for position in range(max_sequence_length):
      positional_encodings.append([])
      for i in range(0, int(embedding_dim / 2)):
        # Even index
        positional_encodings[-1].append(
          math.sin(pos / (10000 ** (2 * i / embedding_dim)))
        )
        # Odd index
        positional_encodings[-1].append(
          math.cos(pos / (10000 ** (2 * i / embedding_dim)))
        )
    result = torch.FloatTensor(
        positional_encodings,
        requires_grad=False
    ).unsqueeze(1)
    assert result.shape == (max_sequence_length, 1, embedding_dim)
    return result

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

    # Add positional encodings
    assert seed.shape[0] <= self.positional_encoding.shape[0]
    assert seed.shape[2] == self.positional_encoding.shape[1]
    seed += self.positional_encoding[:seed.shape[0], :, :]

    assert follow.shape[0] <= self.positional_encoding.shape[0]
    assert follow.shape[2] == self.positional_encoding.shape[1]
    follow += self.positional_encoding[:follow.shape[0], :, :]

    output = self.transformer(
        src=seed,
        tgt=follow,
        src_key_padding_mask=seed_padding_mask,
        tgt_key_padding_mask=follow_padding_mask,
    )
    # output is (F, B, E), however the transformer has been computed

    # V is the size of the "vocab" (total embedding size)
    output = self.predict_output(output)
    # follow is now (F, B, V)

    # produce softmax results across "vocab"
    return self.softmax(output)

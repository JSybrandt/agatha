import torch
import pickle
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
import math
from random import shuffle

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

def get_current_year():
  return datetime.now().year

class AbstractGeneratorTokenizer(object):
  def __init__(
      self,
      tokenizer_model_path:Path,
      extra_data_path:Path,
  ):
    assert tokenizer_model_path.is_file()
    assert extra_data_path.is_file()
    self.sp_processor = spm.SentencePieceProcessor()
    if not self.sp_processor.load(str(tokenizer_model_path)):
      raise ValueError("Invalid model path", tokenizer_model_path)
    with open(extra_data_path, "rb") as f:
      extra_data = pickle.load(f)
    self.mesh_index = extra_data["mesh_index"]
    self.oldest_year = extra_data["oldest_year"]

    self.padding_idx = 0
    self.unknown_idx = 1
    self.sep_idx = 2
    self.mask_idx = 3
    self.special_markers = ["[PAD]", "[UNK]", "[SEP]", "[MASK]"]
    self.special_size = 4
    self.special_start_idx = 0
    self.special_end_idx = self.special_start_idx + self.special_size

    # Sentence Types
    self.sent_type_size = len(INTERESTING_SENTENCE_LABLES)
    self.sent_type_start_idx = self.special_end_idx
    self.sent_type_end_idx = self.sent_type_start_idx + self.sent_type_size
    # Dates
    self.year_size = get_current_year() - self.oldest_year
    self.year_start_idx = self.sent_type_end_idx
    self.year_end_idx = self.year_start_idx + self.year_size
    # Mesh terms
    self.mesh_size = len(self.mesh_index)
    self.mesh_start_idx = self.year_end_idx
    self.mesh_end_idx = self.mesh_start_idx + self.mesh_size
    # Voccab
    self.vocab_start_idx = self.mesh_end_idx
    self.traditional_vocab_size = len(self.sp_processor)
    self.traditional_vocab_end_idx = \
        self.vocab_start_idx + self.traditional_vocab_size
    self.special_vocab_size = 2
    self.vocab_size = self.traditional_vocab_size + self.special_vocab_size
    self.vocab_end_idx = self.vocab_start_idx + self.vocab_size
    # fill in the two extra symbols
    self.start_symbol_idx = self.vocab_end_idx-2
    self.end_symbol_idx = self.vocab_end_idx-1
    self.start_symbol = "[START]"
    self.end_symbol = "[END]"

    self.total_index_size = self.vocab_end_idx


  def __len__(self)->int:
    return self.total_index_size

  def encode_mesh(self, mesh_code:str)->int:
    if mesh_code is None:
      return self.padding_idx
    if self.mesh_index.has_element(mesh_code):
      return self.mesh_index.get_index(mesh_code) + self.mesh_start_idx
    else:
      return self.unknown_idx

  def encode_year(self, year:Optional[int])->int:
    if year is None:
      return self.padding_idx
    if year < self.oldest_year or year > get_current_year():
      return self.unknown_idx
    return year - self.oldest_year + self.year_start_idx

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

  def decode_idx(self, idx:int)->str:
    if 0 <= idx < self.special_end_idx:
      return self.special_markers[idx]
    if self.sent_type_start_idx <= idx < self.sent_type_end_idx:
      return INDEX_TO_SENTENCE_LABEL[idx - self.sent_type_start_idx]
    if self.year_start_idx <= idx < self.year_end_idx:
      return str(idx - self.year_start_idx + self.oldest_year)
    if self.mesh_start_idx <= idx < self.mesh_end_idx:
      return ",".join(self.mesh_index.get_elements(idx - self.mesh_start_idx))
    if self.vocab_start_idx <= idx < self.vocab_end_idx:
      return self.sp_processor.id_to_piece(idx - self.vocab_start_idx)
    return "[INVALID]"

  def decode_text(self, indices:List[int])->str:
    # list of strings
    substrs = []
    # working list of tokens
    working_list = []
    for idx in indices:
      if idx in {self.start_symbol_idx, self.end_symbol_idx}:
        substrs.append(self.sp_processor.decode_ids(working_list))
        working_list = []
        if idx == self.start_symbol_idx:
          substrs.append(self.start_symbol)
        if idx == self.end_symbol_idx:
          substrs.append(self.end_symbol)
      elif self.vocab_start_idx <= idx < self.traditional_vocab_end_idx:
        working_list.append(idx-self.vocab_start_idx)
      else:
        raise ValueError("Invalid idx in text")
    substrs.append(self.sp_processor.decode_ids(working_list))
    return "".join(substrs)

  def encode_context_sequence(
      self,
      year:int,
      mesh_headings:List[str],
  )->List[int]:
    year = self.encode_year(year)
    mesh_headings = [
        self.encode_mesh(m)
        for m in mesh_headings
    ]
    return [year] + mesh_headings + [self.sep_idx]


class AbstractGenerator(torch.nn.Module):
  def __init__(self,
      tokenizer:AbstractGeneratorTokenizer,
      embedding_dim:int,
      max_text_length:int,
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
    self.max_text_length = max_text_length
    self.tokenizer = tokenizer

    self.embeddings = torch.nn.Embedding(
        len(tokenizer),
        embedding_dim,
        padding_idx=tokenizer.padding_idx,
    )

    # Positional encoding is (Max Sequence Length, 1, Embedding Dim)
    self.register_buffer(
        "positional_encoding",
        self.generate_positional_encoding(
          max_text_length=max_text_length,
          embedding_dim=embedding_dim,
        )
    )

    self.transformer = torch.nn.Transformer(
        d_model=embedding_dim,
        nhead=num_attention_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=intermediate_feedforward_dim,
        dropout=intermediate_dropout,
    )

    # This mask has -inf for all values that follow the target input
    # Of size (text, text)
    self.register_buffer(
        "text_attention_mask",
        self.transformer.generate_square_subsequent_mask(max_text_length).t_(),
    )

    self.predicted_text = torch.nn.Linear(
        embedding_dim,
        tokenizer.vocab_size,
    )
    self.predicted_type = torch.nn.Linear(
        embedding_dim,
        tokenizer.sent_type_size,
    )

    self.softmax = torch.nn.LogSoftmax(dim=2)


  def generate_positional_encoding(
      self,
      max_text_length:int,
      embedding_dim:int,
  )->torch.FloatTensor:
    # Dim must be even
    assert embedding_dim % 2 == 0

    # Returns a (seq_len, emb) tensor
    positional_encodings = []
    for pos in range(max_text_length):
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
    ).unsqueeze(1)
    assert result.shape == (max_text_length, 1, embedding_dim)
    return result

  def forward(
      self,
      context:torch.LongTensor,
      text:torch.LongTensor,
      types:torch.LongTensor,
  ):
    # C is the sequence length of context
    # T is the sequence length of text AND types
    # B is the batch size
    assert len(context.shape) == 2
    assert len(text.shape) == 2
    assert len(types.shape) == 2
    # Batch size is consistent
    assert context.shape[1] == text.shape[1] == types.shape[1]
    # T is consistent, and less than the expected max
    assert text.shape[0] == types.shape[0]
    assert text.shape[0] <= self.max_text_length

    # Adds an additional E-sized embedding vector for each long
    context_emb = self.embeddings(context)
    text_emb = self.embeddings(text)
    types_emb = self.embeddings(types)
    # Need to merge text and position
    text_length = text.shape[0]
    txt_typ_pos = text_emb + types_emb + self.positional_encoding[text_length,:,:]

    encoded = self.transformer(
        src=context_emb,
        src_key_padding_mask=(context==self.tokenizer.padding_idx).t_(),
        tgt=txt_typ_pos,
        tgt_key_padding_mask=(text==self.tokenizer.padding_idx).t_(),
        tgt_mask=self.text_attention_mask[:text_length,:text_length],
    )

    predicted_text = self.predicted_text(encoded)
    predicted_type = self.predicted_type(encoded)

    # produce softmax results across "vocab"
    return {
        "text": self.softmax(predicted_text),
        "types": self.softmax(predicted_type),
    }

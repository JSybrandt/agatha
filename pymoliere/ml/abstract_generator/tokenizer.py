import pickle
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
from pymoliere.ml.abstract_generator.sentencepiece_pb2 import SentencePieceText

def get_current_year():
  return datetime.now().year

# http://universaldependencies.org/docs/en/pos/all.html
ALL_POS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "DET", "INTJ", "NOUN", "NUM", "PART",
    "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

# https://universaldependencies.org/u/dep/all.html
ALL_DEP_TAGS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp",
    "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp",
]

# Spacy documentation
ALL_ENTITY_CLASSES = [
    "AMINO_ACID", "ANATOMICAL_SYSTEM", "CANCER", "CELL", "CELLULAR_COMPONENT",
    "DEVELOPING_ANATOMICAL_STRUCTURE", "GENE_OR_GENE_PRODUCT",
    "IMMATERIAL_ANATOMICAL_ENTITY", "MULTI-TISSUE_STRUCTURE", "ORGAN",
    "ORGANISM", "ORGANISM_SUBDIVISION", "SIMPLE_CHEMICAL", "TISSUE",
]

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

    # Dates
    self.year_size = get_current_year() - self.oldest_year
    self.year_start_idx = self.special_end_idx
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

  def decode_idx(self, idx:int)->str:
    if 0 <= idx < self.special_end_idx:
      return self.special_markers[idx]
    if self.year_start_idx <= idx < self.year_end_idx:
      return str(idx - self.year_start_idx + self.oldest_year)
    if self.mesh_start_idx <= idx < self.mesh_end_idx:
      return ",".join(self.mesh_index.get_elements(idx - self.mesh_start_idx))
    if self.vocab_start_idx <= idx < self.vocab_end_idx:
      if idx < self.traditional_vocab_end_idx:
        return self.sp_processor.id_to_piece(idx - self.vocab_start_idx)
      elif idx == self.start_symbol_idx:
        return self.start_symbol
      elif idx == self.end_symbol_idx:
        return self.end_symbol
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

  def decode_context(
      self,
      indices:List[str],
  )->Dict[str, Any]:
    assert indices[-1] == self.sep_idx
    res = {}
    res["year"] = self.decode_idx(indices[0])
    res["mesh_headings"] = [self.decode_idx(i) for i in indices[1:-1]]
    return res

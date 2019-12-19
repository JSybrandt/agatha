import pickle
import sentencepiece as spm
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any, Dict
from pymoliere.ml.abstract_generator.sentencepiece_pb2 import SentencePieceText
from pymoliere.ml.abstract_generator.misc_util import (
    OrderedIndex,
    items_to_ordered_index,
)
from pymoliere.util.misc_util import Record

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
  def __init__(self, tokenizer_model_path:Path, extra_data_path:Path):
    assert tokenizer_model_path.is_file()
    assert extra_data_path.is_file()

    self.subword_tokenizer = spm.SentencePieceProcessor()
    self.subword_tokenizer.load(str(tokenizer_model_path))

    with open(extra_data_path, 'rb') as pickle_file:
      extra_data = pickle.load(pickle_file)
      self.mesh_ord_index = extra_data["mesh_index"]
      self.year_ord_index = items_to_ordered_index(
          range(extra_data["oldest_year"], get_current_year())
      )

    self.pos_ord_index = items_to_ordered_index(ALL_POS_TAGS)
    self.dep_ord_index = items_to_ordered_index(ALL_DEP_TAGS)
    self.ent_ord_index = items_to_ordered_index(ALL_ENTITY_CLASSES)
    # these special characters are used to encode / decode everything
    self.padding_idx = 0
    self.unknown_idx = 1
    self.none_idx = 2
    self.padding_marker = "[PAD]"
    self.unknown_marker = "[UNK]"
    self.none_marker = "[NONE]" # used to indicate when a token is not an entity
    self.base_special_offset = 3
    # these special characters are added to the text encodings
    self.start_idx = 3
    self.end_idx = 4
    self.start_marker = "[START]"
    self.end_marker = "[END]"
    self.text_special_offset = 5

  def _base_encode(self, item:Any, index:OrderedIndex)->int:
    idx = index.get_index(item)
    if idx is None:
      return self.unknown_idx
    return idx + self.base_special_offset

  def _base_decode(self, idx:int, index:OrderedIndex)->Any:
    if idx == self.padding_idx:
      return self.padding_marker
    if idx == self.unknown_idx:
      return self.unknown_marker
    if i == self.none_idx:
      return self.none_marker
    idx -= self.base_special_offset
    values = index[idx] # values is a set
    assert values is not None, "Attempted to decode invalid index"
    assert len(values) == 1, "Ordered indices must return a set of 1 item"
    return next(iter(values))

  def encode_entity_label(self, entity_label:str)->int:
    return self._base_encode(entity_label, self.ent_ord_index)

  def encode_dep(self, dep:str)->int:
    return self._base_encode(dep, self.dep_ord_index)

  def encode_pos(self, pos:str)->int:
    return self._base_encode(pos, self.pos_ord_index)

  def encode_mesh(self, mesh:str)->int:
    return self._base_encode(mesh, self.mesh_ord_index)

  def encode_year(self, year:int)->int:
    return self._base_encode(year, self.year_ord_index)

  def decode_entity_label(self, idx:int)->str:
    return self._base_decode(idx, self.ent_ord_index)

  def decode_dep(self, idx:int)->str:
    return self._base_decode(idx, self.dep_ord_index)

  def decode_pos(self, idx:int)->str:
    return self._base_decode(idx, self.pos_ord_index)

  def decode_mesh(self, idx:int)->str:
    return self._base_decode(idx, self.mesh_ord_index)

  def decode_year(self, idx:int)->int:
    return self._base_decode(idx, self.year_ord_index)

  def len_entity_label(self)->int:
    return len(self.ent_ord_index) + self.base_special_offset

  def len_dep(self)->int:
    return len(self.dep_ord_index) + self.base_special_offset

  def len_pos(self)->int:
    return len(self.pos_ord_index) + self.base_special_offset

  def len_mesh(self)->int:
    return len(self.mesh_ord_index) + self.base_special_offset

  def len_year(self)->int:
    return len(self.year_ord_index) + self.base_special_offset

  def len_text(self)->int:
    return len(self.subword_tokenizer) + self.text_special_offset

  def _parse_text_to_wordpieces(self, text:str)->SentencePieceText:
    result = SentencePieceText()
    proto_text = self.subword_tokenizer.encode_as_serialized_proto(text)
    result.ParseFromString(proto_text)
    return result

  def encode_sentence(
      self,
      sentence:Record,
      is_first:bool=False,
      is_last:bool=False,
  )->Dict[str,List[int]]:
    assert "text" in sentence
    assert "tags" in sentence
    assert "ents" in sentence

    # Index each character
    cha2pos_dep = {}
    cha2ent = {}
    for begin, end, pos, dep in sentence["tags"]:
      pos_idx = self.encode_pos(pos)
      dep_idx = self.encode_dep(dep)
      for char_idx in range(begin, end):
        cha2pos_dep[char_idx] = (pos_idx, dep_idx)
    for begin, end, label in sentence["ents"]:
      ent_idx = self.encode_entity_label(label)
      for char_idx in range(begin, end):
        cha2ent[char_idx] = ent_idx
    def get_first(begin, end, cha2thing, default):
      for cha in range(begin, end):
        thing = cha2thing.get(cha, None)
        if thing is not None:
          return thing
      return default

    text = []
    pos = []
    dep = []
    ent = []
    if is_first:
      text.append(self.start_idx)
      pos.append(self.none_idx)
      dep.append(self.none_idx)
      ent.append(self.none_idx)
    for piece in self._parse_text_to_wordpieces(sentence["text"]).pieces:
      text_idx = piece.id + self.text_special_offset
      ent_idx = get_first(
          piece.begin,
          piece.end,
          cha2ent,
          default=self.none_idx
      )
      pos_idx, dep_idx = get_first(
          piece.begin,
          piece.end,
          cha2pos_dep,
          default=(self.none_idx, self.none_idx)
      )
      text.append(text_idx)
      pos.append(pos_idx)
      dep.append(dep_idx)
      ent.append(ent_idx)
    if is_last:
      text.append(self.end_idx)
      pos.append(self.none_idx)
      dep.append(self.none_idx)
      ent.append(self.none_idx)
    assert len(text) == len(pos) == len(dep) == len(ent)
    return {
        "text": text, "pos": pos, "dep": dep, "ent": ent
    }

  def decode_text(self, ids:List[int])->str:
    substrs = []
    working_ids = []
    for i in ids:
      if i < self.text_special_offset:
        if len(working_ids) > 0:
          substrs.append(self.subword_tokenizer.decode_ids(working_ids))
          working_ids = []
        if i == self.start_idx:
          substrs.append(self.start_marker)
        elif i == self.end_idx:
          substrs.append(self.end_marker)
        elif i == self.unknown_idx:
          substrs.append(self.unknown_marker)
        elif i == self.none_idx:
          substrs.append(self.none_marker)
      else:
        working_ids.append(i-self.text_special_offset)
    substrs.append(self.subword_tokenizer.decode_ids(working_ids))
    return "".join(substrs)

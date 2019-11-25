from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record, iter_to_batches
import torch
from copy import copy
from typing import List
import random
from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer
)

class AbstractWindowGenerator(object):
  def __init__(
      self,
      tokenizer: AbstractGeneratorTokenizer,
      records:List[Record],
      batch_size:int,
      text_size:int,
      return_training_data:bool=True,
      lowercase:bool=True,
      only_first_window_per_abstract:bool=False,
      window_step:int=64,
      minimum_window_size:int=10,
  ):
    self.tokenizer=tokenizer
    self.records = records
    self.batch_size = batch_size
    self.text_size = text_size
    self.maximum_num_batches = int(len(self.records) / self.batch_size)
    self.return_training_data = return_training_data
    self.lowercase = lowercase
    self.only_first_window_per_abstract = only_first_window_per_abstract
    self.window_step = window_step
    self.minimum_window_size = minimum_window_size

  def iterate_batches(
      self
  )->Iterable[Dict[str, torch.tensor]]:
    for batch_data in iter_to_batches(
        self.iterate_data_across_abstracts(),
        self.batch_size,
    ):
      yield {
          field: AbstractWindowGenerator.field_to_long_tensor(batch_data, field)
          for field in batch_data[0]
      }

  def field_to_long_tensor(
      batch_data:List[Dict[str, List[int]]],
      field:str,
  )->torch.LongTensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(d[field]) for d in batch_data],
    )

  def iterate_data_across_abstracts(self)->Iterable[Dict[str, List[int]]]:
    for record in self.records:
      for res in self.iterate_data_within_abstract(record):
        yield res
        if self.only_first_window_per_abstract:
          break

  def iterate_data_within_abstract(
      self,
      abstract:Record,
  )->Iterable[Dict[str, List[int]]]:
    """
    Returns context, text, and shifted
    """

    # Start with [START] Character
    all_text_tokens = [self.tokenizer.start_symbol_idx]

    for text_field in abstract["text_data"]:
      text = text_field["text"]
      if self.lowercase:
        text = text.lower()
      all_text_tokens += self.tokenizer.encode_text(text)
    # End with [END] Character
    all_text_tokens.append(self.tokenizer.end_symbol_idx)

    date = abstract["date"]
    year = int(date.split("-")[0]) if date is not None else -1
    context = self.tokenizer.encode_context_sequence(
        year=year,
        mesh_headings=abstract["mesh_headings"],
    )

    for selection_start in range(0, len(all_text_tokens), self.window_step):
      selection_end = min(
          selection_start + self.text_size,
          len(all_text_tokens)
      )
      if selection_end - selection_start < self.minimum_window_size:
        break

      assert 0 <= selection_start < selection_end <= len(all_text_tokens)

      res = {
          "context": context,
          "text": all_text_tokens[selection_start:selection_end],
      }
      if self.return_training_data:
        res["shifted_text"] = all_text_tokens[selection_start+1:selection_end+1]
        while len(res["shifted_text"]) < len(res["text"]):
          res["shifted_text"].append(self.tokenizer.end_symbol_idx)
      yield res

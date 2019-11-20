from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
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
      device:torch.device,
      batch_size:int,
      text_size:int,
      return_eval_data:bool=False,
      return_training_data:bool=True,
  ):
    self.tokenizer=tokenizer
    self.records = records
    self.batch_size = batch_size
    self.text_size = text_size
    self.maximum_num_batches = int(len(self.records) / self.batch_size)
    self.return_training_data = return_training_data
    self.return_eval_data = return_eval_data
    self.device = device

  def generate(self):
    for idx in range(self.maximum_num_batches):
      yield self.generate_batch(idx)

  def generate_batch(
      self,
      batch_idx:int,
  )->Dict[str, torch.tensor]:
    batch_start_idx = batch_idx * self.batch_size
    assert 0 <= batch_start_idx < len(self.records) - self.batch_size
    context = []
    text = []
    types = []
    shifted_text = []
    shifted_types = []
    remaining_texts = []
    remaining_types = []
    # Make batch
    for record in self.records[batch_start_idx:batch_start_idx+self.batch_size]:
      data = self.abstract_to_training_data(record)
      context.append(torch.LongTensor(data["context"]))
      text.append(torch.LongTensor(data["text"][:-1]))
      types.append(torch.LongTensor(data["types"][:-1]))
      if self.return_training_data:
        shifted_text.append(torch.LongTensor(data["text"][1:]))
        shifted_types.append(torch.LongTensor(data["types"][1:]))
      if self.return_eval_data:
        remaining_texts.append(torch.LongTensor(data["remaining_text"]))
        remaining_types.append(torch.LongTensor(data["remaining_types"]))
    res = {
        "context": torch.nn.utils.rnn.pad_sequence(context),
        "text": torch.nn.utils.rnn.pad_sequence(text),
        "types": torch.nn.utils.rnn.pad_sequence(types),
    }
    if self.return_training_data:
      res["shifted_text"] = torch.nn.utils.rnn.pad_sequence(shifted_text)
      res["shifted_types"] = torch.nn.utils.rnn.pad_sequence(shifted_types)
    if self.return_eval_data:
      res["remaining_text"] = torch.nn.utils.rnn.pad_sequence(remaining_texts)
      res["remaining_types"] = torch.nn.utils.rnn.pad_sequence(remaining_types)
    return {k: v.to(self.device) for k, v in res.items()}

  def abstract_to_training_data(
      self,
      abstract:Record,
  )->Dict[str, List[int]]:
    """
    Returns context, text, and shifted
    """

    all_text_tokens = []
    all_type_tokens = []
    for text_field in abstract["text_data"]:
      tmp_text_tokens = self.tokenizer.encode_text(text_field["text"])
      all_type_tokens += [
          self.tokenizer.encode_sent_type(text_field["type"])
      ] * len(tmp_text_tokens)
      all_text_tokens += tmp_text_tokens

    if len(all_type_tokens) <= self.text_size:
      selection_start = 0
      selection_end = len(all_text_tokens)
    else:
      selection_start = random.randint(0, len(all_type_tokens)-self.text_size-1)
      selection_end = selection_start + self.text_size

    assert selection_start >= 0
    assert selection_end <= len(all_text_tokens)
    assert 0 <= selection_start < selection_end <= len(all_text_tokens)

    context = self.tokenizer.encode_context_sequence(
        year=int(abstract["date"].split("-")[0]),
        authors=abstract["authors"],
        mesh_headings=abstract["mesh_headings"],
    )
    text = all_text_tokens[selection_start:selection_end]
    types = all_type_tokens[selection_start:selection_end]
    res = {
        "context": context,
        "text": text,
        "types": types,
    }
    if self.return_eval_data:
      res["remaining_text"] = all_text_tokens[selection_end:]
      res["remaining_types"] = all_type_tokens[selection_end:]
    return res

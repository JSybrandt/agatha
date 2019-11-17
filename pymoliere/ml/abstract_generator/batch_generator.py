from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
import torch
from copy import copy
from typing import List
import random
from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer
)
import torch.multiprocessing as mp
import queue

class AbstractWindowGenerator(object):
  def __init__(
      self,
      records:List[Record],
      num_workers:int,
      queue_size:int,
      device:torch.device,
      **worker_kwargs
  ):
    assert num_workers > 0
    assert queue_size > 0
    self.queue = mp.Queue(queue_size)
    self.device = device
    def get_worker(idx):
      part_size = int(len(records) / num_workers)
      records_start = idx * part_size
      records_end = min(len(records), records_start + part_size)
      return _AbstractWindowGeneratorWorker(
          queue=self.queue,
          records=records[records_start:records_end],
          **worker_kwargs
      )
    self.processes = [get_worker(i) for i in range(num_workers)]

  def generate(self):
    for p in self.processes:
      p.start()
    try:
      while True:
        data =  self.queue.get(block=True, timeout=1)
        for key in data:
          data[key] = data[key].to(self.device)
        yield data
    except queue.Empty:
      print("Empty queue")
      pass
    for p in self.processes:
      p.terminate()

  def __del__(self):
    for p in self.processes:
      p.terminate()



class _AbstractWindowGeneratorWorker(mp.Process):
  def __init__(
      self,
      queue:mp.Queue,
      records:List[Record],
      batch_size:int,
      text_size:int,
      **tokenizer_kwargs,
  ):
    """
    Generates batches asynchronously and infinitely.
    Parameters:
      - tokenizer: the object used to process abstracts
      - records: the dict objects containing abstract text and metadata
      - batch_size: the number of examples per iteration
    """
    super(_AbstractWindowGeneratorWorker, self).__init__()
    # Can't pickle the tokenizer, so we need to construct one per-process
    self.tokenizer = AbstractGeneratorTokenizer(**tokenizer_kwargs)
    self.records = records
    self.batch_size = batch_size
    self.text_size = text_size
    self.num_batches = int(len(self.records) / self.batch_size)
    self.queue = queue


  def run(self):
    for batch_idx in range(self.num_batches):
      self.queue.put(self.generate_batch(batch_idx))


  def generate_batch(self, batch_idx:int)->Dict[str, torch.tensor]:
    batch_start_idx = batch_idx * self.batch_size
    assert 0 <= batch_start_idx < len(self.records) - self.batch_size
    context = []
    text = []
    types = []
    shifted_text = []
    shifted_types = []
    # Make batch
    for record in self.records[batch_start_idx:batch_start_idx+self.batch_size]:
      data = self.abstract_to_training_data(record)
      context.append(torch.LongTensor(data["context"]))
      text.append(torch.LongTensor(data["text"][:-1]))
      types.append(torch.LongTensor(data["types"][:-1]))
      # we are going to set the domain of shifted indices to the smaller ranges
      shifted_text.append(
          torch.LongTensor(data["text"][1:])\
              -self.tokenizer.vocab_start_idx

      )
      shifted_types.append(
          torch.LongTensor(data["types"][1:])\
              -self.tokenizer.sent_type_start_idx
      )

    # pad
    context = torch.nn.utils.rnn.pad_sequence(context)
    text = torch.nn.utils.rnn.pad_sequence(text)
    types = torch.nn.utils.rnn.pad_sequence(types)
    shifted_text = torch.nn.utils.rnn.pad_sequence(shifted_text)
    shifted_types = torch.nn.utils.rnn.pad_sequence(shifted_types)
    return {
        "context": context,
        "text": text,
        "types": types,
        "shifted_text": shifted_text,
        "shifted_types": shifted_types,
    }


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
    return {
        "context": context,
        "text": text,
        "types": types,
    }

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
    while not self.queue.empty():
      kwargs, target =  self.queue.get(block=True, timeout=3)
      for key in kwargs:
        kwargs[key] = kwargs[key].to(self.device)
      target = target.to(self.device)
      yield kwargs, target
    for p in self.processes:
      p.join()


class _AbstractWindowGeneratorWorker(mp.Process):
  def __init__(
      self,
      queue:mp.Queue,
      records:List[Record],
      batch_size:int,
      seed_text_size:int,
      follow_text_size:int,
      difficulty:float,
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
    assert 0 < difficulty < 1
    # Can't pickle the tokenizer, so we need to construct one per-process
    self.tokenizer = AbstractGeneratorTokenizer(**tokenizer_kwargs)
    self.records = records
    self.batch_size = batch_size
    self.seed_text_size = seed_text_size
    self.follow_text_size = follow_text_size
    self.difficulty = difficulty
    self.num_batches = int(len(self.records) / self.batch_size)
    self.queue = queue

  def run(self):
    for batch_idx in range(self.num_batches):
      self.queue.put(self.generate_batch(batch_idx))

  def generate_batch(self, batch_idx:int)->Tuple[Dict[str, torch.tensor], torch.tensor]:
    batch_start_idx = batch_idx * self.batch_size
    assert 0 <= batch_start_idx < len(self.records) - self.batch_size
    seed = []
    follow = []
    # Make batch
    for record in self.records[batch_start_idx:batch_start_idx+self.batch_size]:
      s, f = self.abstract_to_training_data(record)
      seed.append(torch.LongTensor(s))
      follow.append(torch.LongTensor(f))
    # pad
    seed = torch.nn.utils.rnn.pad_sequence(seed)
    follow = torch.nn.utils.rnn.pad_sequence(follow)
    # Move to dev
    # Training masks tokens
    corrupted = follow.clone()
    corrupted[
        torch.rand_like(corrupted, dtype=torch.float32)
        < self.difficulty
    ] = self.tokenizer.mask_idx
    return {"seed": seed, "follow": corrupted}, follow


  def abstract_to_training_data(
      self,
      abstract:Record,
  )->Tuple[List[int], List[int]]:
    """
    Given an abstract, randomly select a seed and target.
    Process these text windows using the tokenizer.
    Return seed and target token sequences
    """

    year = int(abstract["date"].split("-")[0])
    mesh_headings = abstract["mesh_headings"]
    authors = abstract["authors"]

    total_tokens = []
    total_types = []
    for text_field in abstract["text_data"]:
      ab_tokens = self.tokenizer.encode_text(text_field["text"])
      ab_types = [text_field["type"]] * len(ab_tokens)
      total_tokens += ab_tokens
      total_types += ab_types

    # pick a window of variable size
    # we're going to look at windows of size n/2 -> n
    # Note, the local [seed/follow]_text_size are randomly chosen based off
    # the object's versions.
    seed_text_size = random.randint(
        int(self.seed_text_size/2),
        self.seed_text_size
    )
    follow_text_size = random.randint(
        int(self.follow_text_size/2),
        self.follow_text_size
    )
    seed_to_window_ratio = seed_text_size / (seed_text_size + follow_text_size)
    # we would like a total selection of this size
    window_size = min(
        seed_text_size + follow_text_size,  # desired
        len(total_tokens), # largest possible
    )
    assert window_size > 1
    seed_text_size = int(seed_to_window_ratio * window_size)
    follow_text_size = window_size - seed_text_size
    assert seed_text_size >= 1
    assert follow_text_size >= 1

    assert seed_text_size + follow_text_size == window_size
    assert window_size <= len(total_tokens)

    seed_selection_start = random.randint(0, len(total_tokens)-window_size)
    follow_selection_start = seed_selection_start + seed_text_size
    follow_selection_end = follow_selection_start + follow_text_size

    seed = self.tokenizer.encode_all(
        max_text_length=seed_text_size,
        year=year,
        authors=authors,
        mesh_headings=mesh_headings,
        start_sentence_type=total_types[seed_selection_start],
        end_sentence_type=total_types[follow_selection_start-1],
        text_indices=total_tokens[seed_selection_start:follow_selection_start]
    )
    follow = self.tokenizer.encode_all(
        max_text_length=follow_text_size,
        year=year,
        authors=authors,
        mesh_headings=mesh_headings,
        start_sentence_type=total_types[follow_selection_start],
        end_sentence_type=total_types[follow_selection_start-1],
        text_indices=total_tokens[follow_selection_start:follow_selection_end]
    )
    return seed, follow

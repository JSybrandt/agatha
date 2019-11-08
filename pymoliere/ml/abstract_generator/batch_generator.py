from typing import Dict, Any, Iterable, Tuple
from pymoliere.util.misc_util import Record
import torch
from copy import copy
from typing import List
import random
from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer
)

try:
  from nlgeval import NLGEval
except ImportError:
  # This is a heavy dependency, and we don't want to worry all users with it.
  pass

class AbstractWindowGenerator(object):
  def __init__(
      self,
      tokenizer:AbstractGeneratorTokenizer,
      records:List[Record],
      batch_size:int,
      required_author_count:int,
      required_mesh_count:int,
      seed_text_size:int,
      follow_text_size:int,
  ):
    """
    Generates batches asynchronously and infinitely.
    Parameters:
      - tokenizer: the object used to process abstracts
      - records: the dict objects containing abstract text and metadata
      - batch_size: the number of examples per iteration
    """
    self.tokenizer = tokenizer
    self.records = records
    self.batch_size = batch_size
    self.required_author_count = required_author_count
    self.required_mesh_count = required_mesh_count
    self.seed_text_size = seed_text_size
    self.follow_text_size = follow_text_size

  def __iter__(self):
    return self

  def __next__(self):
    seeds, follows = self.generate_batch()
    seeds = list(map(torch.LongTensor, seeds))
    follows = list(map(torch.LongTensor, follows))
    seeds = torch.nn.utils.rnn.pad_sequence(seeds)
    follows = torch.nn.utils.rnn.pad_sequence(follows)
    return seeds, follows

  def generate_batch(self)->Tuple[torch.LongTensor, torch.LongTensor]:
    seeds = []
    follows = []
    for _ in range(self.batch_size):
      record = random.choice(self.records)
      seed, follow = self.abstract_to_training_data(record)
      seeds.append(seed)
      follows.append(follow)
    return seeds, follows


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
        required_author_count=self.required_author_count,
        required_mesh_count=self.required_mesh_count,
        max_text_length=seed_text_size,
        year=year,
        authors=authors,
        mesh_headings=mesh_headings,
        start_sentence_type=total_types[seed_selection_start],
        end_sentence_type=total_types[follow_selection_start-1],
        text_indices=total_tokens[seed_selection_start:follow_selection_start]
    )
    follow = self.tokenizer.encode_all(
        required_author_count=self.required_author_count,
        required_mesh_count=self.required_mesh_count,
        max_text_length=follow_text_size,
        year=year,
        authors=authors,
        mesh_headings=mesh_headings,
        start_sentence_type=total_types[follow_selection_start],
        end_sentence_type=total_types[follow_selection_start-1],
        text_indices=total_tokens[follow_selection_start:follow_selection_end]
    )
    return seed, follow

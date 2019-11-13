from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer,
)
from pymoliere.util.misc_util import Record, iter_to_batches
import torch
from typing import Tuple, List, Iterable
try:
  from nlgeval import NLGEval
except ImportError:
 # This is a heavy dependency, and we don't want to worry all users with it.
 pass


class GenerationEvalBatchGenerator(object):
  """
  The goal here is to generate predictable inputs for the model, and a range of
  reference outputs that could be valid to generate.
  """

  def __init__(
      self,
      tokenizer:AbstractGeneratorTokenizer,
      records:List[Record],
      device:torch.device,
      batch_size:int,
      seed_text_size:int,
      follow_text_size:int,
      reference_step_size:int,
      max_num_references:int,
  ):
    """

    We're going to loop through each abstract, and pull out all possible
    non-overlapping seeds.  Then, for each seed, we're going to produce a mask,
    as well as a list of references that could be valid follow-ups to the seed.
    The goal will be to generate one of the references given (seed, [mask]) as
    input.

    reference_step_size: the number of tokens to jump between the start of each reference
    max_num_references: the maximal number of references to produce
    """

    self.tokenizer = tokenizer
    self.records = records
    self.device = device
    self.batch_size = batch_size
    self.seed_text_size = seed_text_size
    self.follow_text_size = follow_text_size
    self.reference_step_size = reference_step_size
    self.max_num_references = max_num_references
    self.seed_to_window_ratio = \
        seed_text_size / (seed_text_size + follow_text_size)
    self.window_size = seed_text_size + follow_text_size

  def generate(
      self
  )->Iterable[Tuple[torch.Tensor, torch.Tensor, List[List[str]]]]:
    for batch in iter_to_batches(self.iterate_all_windows(), self.batch_size):
      seed = torch.nn.utils.rnn.pad_sequence([s for s, f, rs in batch])
      follow = torch.nn.utils.rnn.pad_sequence([f for s, f, rs in batch])
      references = [rs for s, f, rs in batch]
      yield seed.to(self.device), follow.to(self.device), references

  def iterate_all_windows(
      self
  )->Iterable[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
    for record in self.records:
      for seed, follow, references in self.iterate_windows_in_abstract(record):
        yield seed, follow, references

  def iterate_windows_in_abstract(
      self,
      abstract:Record
  )->Iterable[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
    """
    Returns seed, original follow, and list of references
    """
    # metadata
    year = int(abstract["date"].split("-")[0])
    mesh_headings = abstract["mesh_headings"]
    authors = abstract["authors"]

    # list of tokens and types at each token
    total_tokens = []
    total_types = []
    for text_field in abstract["text_data"]:
      ab_tokens = self.tokenizer.encode_text(text_field["text"])
      ab_types = [text_field["type"]] * len(ab_tokens)
      total_tokens += ab_tokens
      total_types += ab_types

    for start_idx in range(0, len(abstract), self.seed_text_size):
      window_size = min(
          self.window_size,
          len(total_tokens) - start_idx
      )
      seed_text_size = int(window_size * self.seed_to_window_ratio)
      follow_text_size = window_size - seed_text_size

      seed_end = start_idx + seed_text_size
      window_end = start_idx + window_size

      seed = self.tokenizer.encode_all(
          max_text_length=self.seed_text_size,
          year=year,
          authors=authors,
          mesh_headings=mesh_headings,
          start_sentence_type=total_types[start_idx],
          end_sentence_type=total_types[seed_end-1],
          text_indices=total_tokens[start_idx:seed_end]
      )
      follow = self.tokenizer.encode_all(
          max_text_length=self.follow_text_size,
          year=year,
          authors=authors,
          mesh_headings=mesh_headings,
          start_sentence_type=total_types[seed_end],
          end_sentence_type=total_types[window_end-1],
          text_indices=total_tokens[seed_end:window_end]
      )
      references = [
          self.tokenizer.decode_text(
            total_tokens[ref_start_idx:(ref_start_idx+follow_text_size)]
          )
          for ref_start_idx in list(
            range(
              seed_end,
              len(total_tokens)-follow_text_size,
              self.reference_step_size
            )
          )[:self.max_num_references]
      ]
      yield torch.LongTensor(seed), torch.LongTensor(follow), references

import torch
from torch.utils.data import Dataset
from agatha.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
import random
from typing import Dict, Tuple, List, Any, Set

def shift_text_features_for_training(
    batch:Dict[str, torch.LongTensor]
)-> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor]]:
  model_in_keys = {'text', 'year', 'mesh'}
  expected_keys = {'text', 'pos', 'dep', 'ent'}
  for key in model_in_keys.union(expected_keys):
    assert key in batch, f"Batch missing {key}"
  expected = {
      # remember, that batch is 2nd dim, and seq is first
      # remove first token
      key: batch[key][1:, :]
      for key in expected_keys
  }
  model_in = {
      key: batch[key]
      for key in model_in_keys
  }
  # Crop the last element from the model input text
  model_in["text"] = model_in["text"][:-1]
  return (model_in, expected)


def collate_encoded_abstracts(
    batch:List[Dict[str,List[int]]],
    key_subset:Set[str]=None,
)->Dict[str, torch.LongTensor]:
  needed_keys = {'text', 'pos', 'dep', 'ent', 'year', 'mesh'}

  if key_subset is not None:
    # Assert the subset is actually a subset
    assert key_subset.intersection(needed_keys) == key_subset
    needed_keys = key_subset

  for needed_key in needed_keys:
    for b in batch:
      assert needed_key in b, f"Encoded abstract missing {needed_key}"
  return {
      key: torch.nn.utils.rnn.pad_sequence(
          [torch.LongTensor(b[key]) for b in batch]
      )
      for key in needed_keys
  }

class EncodedAbstracts(Dataset):
  def __init__(
      self,
      abstract_ds:Dataset,
      tokenizer_kwargs:Dict[str, Any],
      max_text_length:int,
      max_mesh_length:int,
      title_only=False,
      return_abstract=False,
  ):
    super(EncodedAbstracts, self).__init__()
    self.abstract_ds = abstract_ds
    self.title_only = title_only
    self.return_abstract = return_abstract
    self.max_text_length = max_text_length
    self.max_mesh_length = max_mesh_length
    self.tokenizer_kwargs = tokenizer_kwargs

  # These three functions are needed to not pickle the tokenizer
  def __getstate__(self):
    # We need to exclude the tokenizer
    state = self.__dict__.copy()
    if "tokenizer" in state:
      del state["tokenizer"]
    return state

  def __setstate(self, state):
    self.__dict__.update(state)
    self.init_tokenizer()

  def init_tokenizer(self):
    if not hasattr(self, "tokenizer"):
      self.tokenizer = AbstractGeneratorTokenizer(
          **self.tokenizer_kwargs
      )

  def __len__(self):
    return len(self.abstract_ds)

  def __getitem__(self, data_idx):
    self.init_tokenizer()
    abstract = self.abstract_ds[data_idx]
    # Asserts
    for needed_key in ['year', 'mesh_headings', 'sentences']:
      assert needed_key in abstract, f"Abstract missing {needed_key}"
    for sent in abstract["sentences"]:
      for needed_key in ["text", "type", "tags", "ents"]:
        assert needed_key in sent, f"Sentence missing {needed_key}"
    # Pick a start sentence
    sents = abstract["sentences"]
    if self.title_only:
      # If we're only outputting titles, just subset to that
      title_sents = list(filter(lambda s: s["type"]=="title", sents))
      if len(title_sents) == 0:
        # promote first sentence to be title
        title_sents = [sents[0]]
        title_sents[0]["type"] == "title"
      sents = title_sents
      selected_sent_idx = 0
    else:
      selected_sent_idx = random.randint(0, len(sents)-1)
    # encode text starting with the selected sentence to max length
    result = None
    for sent_idx in range(selected_sent_idx, len(sents)):
      is_first = sent_idx==0
      # in the case we're only getting the title, drop the [END]
      is_last = (sent_idx == len(sents)-1) and not self.title_only
      tmp_tokens = self.tokenizer.encode_sentence(
          sents[sent_idx], is_first, is_last
      )
      assert "text" in tmp_tokens
      if result is None:
        result = tmp_tokens
      else:
        for key in result:
          result[key] += tmp_tokens[key]
      if len(result["text"]) >= self.max_text_length:
        break

    # truncate what we got to max text length
    for key in result:
      result[key] = result[key][:self.max_text_length]
    assert "year" not in result
    result["year"] = [self.tokenizer.encode_year(abstract["year"])]
    assert "mesh" not in result
    result["mesh"] = [
        self.tokenizer.encode_mesh(m)
        for m in abstract["mesh_headings"][:self.max_mesh_length]
    ]
    if self.return_abstract:
      return (result, abstract)
    else:
      return result

from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGenerator,
)
from copy import deepcopy
import json
import numpy as np
from pathlib import Path
import pickle
import pygsheets
from pymoliere.config import config_pb2 as cpb
from pymoliere.construct import text_util
from pymoliere.ml.abstract_generator import datasets
from pymoliere.ml.abstract_generator.path_util import get_paths
from pymoliere.ml.abstract_generator.tokenizer import AbstractGeneratorTokenizer
from pymoliere.util.misc_util import Record
import re
import torch
from typing import Tuple, List, Iterable, Dict, Any
try:
  from nlgeval import NLGEval
except ImportError:
 # This is a heavy dependency, and we don't want to worry all users with it.
 pass

class MultiLogger():
  def __init__(self, config:cpb.AbstractGeneratorConfig):
    self.log_to_gsheets = config.HasField("gsheets_api_cred")
    if self.log_to_gsheets:
      assert Path(config.gsheets_api_cred).is_file()
      self.gsheets_client = pygsheets.authorize(client_secret=config.gsheets_api_cred)
      self.gsheets_sheet = self.gsheets_client.create(config.gsheets_title)
      self.gsheets_worksheet = self.gsheets_sheet.sheet1
      print("Logging to google sheets:", self.gsheets_sheet.url)
    self.log_to_console = True
    self.column_order = None
    self.row_idx = -1

  def log_row(self, vals:Dict[str, Any])->None:
    if self.column_order is None:
      self.column_order = [k for k in vals]
      # Log the header
      self.log_row({c:c for c in self.column_order})
    # get a new row
    self.row_idx += 1
    if self.log_to_console:
      self._log_console(vals)
    if self.log_to_gsheets:
      self._log_gsheets(vals)

  def _log_console(vals: Dict[str, Any])->None:
    assert self.column_order is not None, "Must init column_order first"
    assert self.log_to_console, "_log_console called when not log_to_console"
    fmtrow = "\t".join([
      str(vals[col]) if col in vals else "N/A"
      for col in self.column_order
    ])
    print(f"{self.row_idx}|{fmtrow}")

  def _log_gsheets(vals:Dict[str, Any])->None:
    assert self.column_order is not None, "Must init column_order first"
    assert self.log_to_gsheets, "_log_gsheets called when not log_to_gsheets"
    values = [vals[col] if col in vals else None for col in self.column_order]
    self.gsheets_worksheet.add_rows(
        row=self.row_idx,
        values=values
    )


def evaluate(
    config:cpb.AbstractGeneratorConfig,
    num_trials:int=1,
    gen_whole_abstract:bool=True,
    skip_metrics:bool=True,
):

  multilogger = MultiLogger(config)

  assert config.HasField("restore_from_checkpoint"), \
      "Must supply restore_from_checkpoint config"
  paths = get_paths(config)

  testing_data_dir = paths["model_ckpt_dir"].joinpath("testing_data")
  assert testing_data_dir.is_dir()

  model = AbstractGenerator.load_from_checkpoint(config.restore_from_checkpoint)
  model.init_tokenizer()
  model.freeze()
  model.eval()

  for test_pkl in testing_data_dir.glob("*.pkl"):
    with open(test_pkl, "rb") as pkl_file:
      abstracts = pickle.load(pkl_file)
      encoder = datasets.EncodedAbstracts(
          abstract_ds=abstracts,
          tokenizer_kwargs=model.hparams.tokenizer_kwargs,
          max_text_length=model.hparams.max_text_length,
          max_mesh_length=model.hparams.max_text_length-1,
          title_only=True,
          return_abstract=True,
      )
      loader = torch.utils.data.DataLoader(
          dataset=encoder,
          batch_size=1,
          collate_fn=collate_for_generation,
          shuffle=True,
      )

      # loader typically returns a list, but we set this to batch of 1
      for model_in, (abstract,) in loader:
        reference_sentences = [
            sent["text"]
            for sent in abstract["sentences"]
            if sent["type"] != "title"
        ]
        if len(reference_sentences) == 0:
          continue
        title = " ".join(
            [s["text"] for s in abstract["sentences"] if s["type"] == "title"]
        )
        for trial_idx in range(num_trials):
          new_sentence = generate_new_text(
              model,
              model_in,
              gen_whole_abstract,
              min_size=3,
          )
          if skip_metrics:
            print("PMID:", abstract["pmid"])
            print("TITLE:", title)
            print("GEN:", new_sentence)
            print("---")
          if skip_metrics:
            metrics = {}
          else:
            metrics = get_nlg_eval().compute_individual_metrics(
                reference_sentences,
                new_sentence,
            )
            metrics = {k: float(v) for k, v in metrics.items()}
          metrics["generated_text"] = new_sentence
          metrics["pmid"] = abstract["pmid"]
          metrics["title"] = title
          multilogger.log_row(metrics)


def collate_for_generation(batch):
  assert isinstance(batch[0], tuple), "Generation requires return_abstract"
  tokens = [b[0] for b in batch]
  abstracts = [b[1] for b in batch]
  model_in = datasets.collate_encoded_abstracts(
      tokens,
      key_subset={"text", "year", "mesh"}
  )
  return model_in, abstracts

def get_nlg_eval():
  if not hasattr(get_nlg_eval, "nlg_eval"):
    print("Loading eval data (first time only)")
    get_nlg_eval.nlg_eval = NLGEval()
  return get_nlg_eval.nlg_eval


def generate_new_text_tokens(
  model:AbstractGenerator,
  model_in:Dict[str, torch.Tensor]
)->Iterable[Tuple[str, str]]:
  assert "text" in model_in
  assert model_in["text"].shape[1] == 1, "Only support batch size 1 currently."

  while True:
    predictions = model(**model_in)

    # Remember, we're using logsoftmax as output
    word_probabilities = np.exp(
        predictions["text"][-1, 0, :].detach().cpu().numpy()
    )

    choices = []
    probs = []
    for idx, prob in enumerate(word_probabilities):
      if prob > 0.001:
        choices.append(idx)
        probs.append(prob)

    probs = np.array(probs, dtype=np.float32)
    probs /= probs.sum()

    new_word = int(np.random.choice(choices, p=probs))
    yield new_word

    def add_and_shift(tensor, new_element):
      l = tensor.flatten().tolist()
      l.append(new_element)
      if len(l) >= model.hparams.max_text_length:
        l = l[-model.hparams.max_text_length+1:]
      return torch.LongTensor(l).unsqueeze(1).to(tensor.device)

    model_in["text"] = add_and_shift(model_in["text"], new_word)

def generate_new_text(
    model:AbstractGenerator,
    model_in:Dict[str, torch.Tensor],
    gen_whole_abstract:bool=False,
    min_size:int=None,
    max_size:int=None,
)->str:
  model.init_tokenizer()
  res = []
  # will run forever if allowed to
  for new_token in generate_new_text_tokens(model, model_in):
    res.append(new_token)
    partial_text = model.tokenizer.decode_text([new_token])
    if min_size is not None and len(res) < min_size:
      continue
    if max_size is not None and len(res) >= max_size:
      break
    if partial_text.endswith(".") and not gen_whole_abstract:
      break
    if new_token == model.tokenizer.end_idx:
      break
  # Don't actually want to see the end token
  if res[-1] == model.tokenizer.end_idx:
    res = res[:-1]
  return model.tokenizer.decode_text(res)


def name_thy_self(config:cpb.AbstractGeneratorConfig)->str:
  assert config.HasField("restore_from_checkpoint"), \
      "Must supply restore_from_checkpoint config"
  paths = get_paths(config)
  model = AbstractGenerator.load_from_checkpoint(config.restore_from_checkpoint)
  model.init_tokenizer()
  model.freeze()
  model.eval()

  text = """
    Medical Hypothesis Generation via. Conditional Abstract Generation. In
    this work, we present a variant of GPT-2 that incorporates medical domain
    knowledge. This system, which we have named py
  """
  text = re.sub(r"\s+", " ", text)
  text = text.strip()

  abstract = dict(
      pmid=0000,
      year=2019,
      mesh_headings=[],
      sentences=[dict(
        type="title",
        text=text,
        tags=[],
        ents=[],
      ), dict(
        type="abstract:raw",
        text="Discard this.",
        tags=[],
        ents=[],
      )]
  )

  encoder = datasets.EncodedAbstracts(
      abstract_ds=[abstract],
      tokenizer_kwargs=model.hparams.tokenizer_kwargs,
      max_text_length=model.hparams.max_text_length,
      max_mesh_length=model.hparams.max_text_length-1,
      title_only=True,
      return_abstract=True,
  )

  loader = torch.utils.data.DataLoader(
      dataset=encoder,
      batch_size=1,
      collate_fn=collate_for_generation,
  )

  for model_in, abstract in loader:
    new_sentence = generate_new_text(
        model,
        model_in,
        min_size=3,
        max_size=10,
    )
    print(new_sentence)


"""
def calculate_first_sentence_perplexity(
    model:AbstractGenerator,
    tokenizer:AbstractGeneratorTokenizer,
    context:torch.LongTensor,
    initial_text:torch.LongTensor,
    evaluated_text:torch.LongTensor
)->float:
  assert len(context.shape) == len(initial_text.shape) \
      == len(evaluated_text.shape) == 2
  # Only handling batch size 1 right now
  assert context.shape[1] == initial_text.shape[1] \
      == evaluated_text.shape[1] == 1

  # input text is both values merged
  all_text = torch.cat((initial_text, evaluated_text))
  # predictions is size (len(all_text), 1, vocab_size)
  predictions = model(context, all_text)["text"].detach()
  assert predictions.shape[0] == all_text.shape[0]
  # Multiplying resulting probs here
  product = 1
  # iterate through the evaluated component
  for prediction_idx in range(len(initial_text), len(all_text)):
    # this is the given token in the expected section
    expected_token = \
        int(all_text[prediction_idx, 0]) - tokenizer.vocab_start_idx

    # We predicted this probability from the n-1'th position
    # Remember that our model outputs log-probabilities
    log_prob = float(predictions[prediction_idx-1, 0, expected_token])
    product *= (1 / np.exp(log_prob))
  return product ** (1 / len(evaluated_text))
"""

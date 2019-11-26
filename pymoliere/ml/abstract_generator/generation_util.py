from pymoliere.ml.abstract_generator.abstract_generator import (
    AbstractGeneratorTokenizer,
    AbstractGenerator,
)
from pymoliere.ml.abstract_generator.batch_generator import (
    AbstractWindowGenerator,
)
from pymoliere.util.misc_util import Record, iter_to_batches
from pymoliere.construct import text_util
import torch
from typing import Tuple, List, Iterable, Dict, Any
from copy import deepcopy
try:
  from nlgeval import NLGEval
except ImportError:
 # This is a heavy dependency, and we don't want to worry all users with it.
 pass

def evaluate_model_on_abstract(
    abstract:Record,
    tokenizer:AbstractGeneratorTokenizer,
    model:AbstractGenerator,
    text_length:int,
    device:torch.device,
)->Dict[str, Any]:

  title_only = deepcopy(abstract)
  title_only["text_data"] = [title_only["text_data"][0]]
  assert title_only["text_data"][0]["type"] == "title"

  batch_generator = AbstractWindowGenerator(
      tokenizer=tokenizer,
      records=[title_only],
      batch_size=1,
      text_size=text_length,
      return_training_data=False,
      only_first_window_per_abstract=True,
  )
  model_input = next(batch_generator.iterate_batches())

  # Remove the end token from the title, which should enable the model to keep
  # going
  assert "text" in model_input
  assert "context" in model_input
  assert model_input["text"][-1] == tokenizer.end_symbol_idx
  model_input["text"] = model_input["text"][:-1]

  text_generator = generate_new_text(
      model=model,
      tokenizer=tokenizer,
      context=(
        torch.LongTensor(model_input["context"]).to(device)
      ),
      text=(
        torch.LongTensor(model_input["text"]).to(device)
      ),
  )

  generated_indices = []
  for next_idx in text_generator:
    generated_indices.append(next_idx)
    next_token = tokenizer.decode_idx(next_idx)
    if (
        next_token == tokenizer.end_symbol
        or next_token[-1] == "."
        or len(generated_indices) >= 100
    ):
      break

  if hasattr(evaluate_model_on_abstract, "nlg_eval"):
    nlg_eval = evaluate_model_on_abstract.nlg_eval
  else:
    nlg_eval = evaluate_model_on_abstract.nlg_eval = NLGEval()

  generated_sentence = tokenizer.decode_text(generated_indices)
  reference_sentences = [
      s["sent_text"]
      for s in text_util.split_sentences([abstract])[1:]
  ]

  metrics = nlg_eval.compute_individual_metrics(
      reference_sentences,
      generated_sentence
  )

  metrics["pmid"] = abstract["pmid"]
  metrics["title"] = title_only["text_data"][0]["text"]
  metrics["references"] = reference_sentences
  metrics["generated_text"] = generated_sentence
  return metrics


def generate_new_text(
  model:AbstractGenerator,
  tokenizer:AbstractGeneratorTokenizer,
  context:torch.LongTensor,
  text:torch.LongTensor=None,
)->Iterable[Tuple[str, str]]:
  # Can give both or neither
  if text is None:
    text = (
        torch.LongTensor([tokenizer.start_symbol_idx])
        .unsqueeze(1)
        .to(context.device)
    )

  # Only supporting batch size 1 for now
  assert context.shape[1] == text.shape[1] == 1

  while True:
    predictions = model(context, text)

    new_word = predictions["text"][-1, 0, :].argmax() \
        + tokenizer.vocab_start_idx

    yield int(new_word)

    def add_or_shift(tensor, new_element):
      l = tensor.flatten().tolist()
      l.append(new_element)
      if len(l) >= model.max_text_length:
        l = l[-model.max_text_length+1:]
      return torch.LongTensor(l).unsqueeze(1).to(tensor.device)

    text = add_or_shift(text, int(new_word))

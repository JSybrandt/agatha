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
import numpy as np
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
    lowercase:bool,
)->Dict[str, Any]:

  if lowercase:
    abstract["text_data"] = [
        {"text": t["text"].lower(), "type": t["type"]}
        for t in abstract["text_data"]
    ]

  title_only = deepcopy(abstract)
  title_only["text_data"] = [title_only["text_data"][0]]
  assert title_only["text_data"][0]["type"] == "title"

  reference_sentences = [
      s["sent_text"]
      for s in text_util.split_sentences([abstract])[1:]
  ]

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

  context = torch.LongTensor(model_input["context"]).to(device)
  title = torch.LongTensor(model_input["text"]).to(device)
  first_sentence = torch.LongTensor(
      tokenizer.encode_text(reference_sentences[0])
  ).unsqueeze(1).to(device)

  perplexity_of_first_sentence = calculate_first_sentence_perplexity(
      model=model,
      tokenizer=tokenizer,
      context=context,
      initial_text=title,
      evaluated_text=first_sentence,
  )

  print("Perp:", perplexity_of_first_sentence)

  text_generator = generate_new_text(
      model=model,
      tokenizer=tokenizer,
      context=context,
      text=title,
  )

  best_generated_result = None
  for trial_idx in range(10):
    generated_indices = []
    for next_idx in text_generator:
      generated_indices.append(next_idx)
      next_token = tokenizer.decode_idx(next_idx)
      if (
          len(generated_indices) > 5
          and next_token == tokenizer.end_symbol
          or next_token[-1] == "."
          or len(generated_indices) >= 100
      ):
        break

    if hasattr(evaluate_model_on_abstract, "nlg_eval"):
      nlg_eval = evaluate_model_on_abstract.nlg_eval
    else:
      print("Loading eval data (first time only)")
      nlg_eval = evaluate_model_on_abstract.nlg_eval = NLGEval()

    generated_sentence = tokenizer.decode_text(generated_indices)

    metrics = nlg_eval.compute_individual_metrics(
        reference_sentences,
        generated_sentence
    )
    metrics = {k: float(v) for k, v in metrics.items()}
    metrics["generated_text"] = generated_sentence

    if (
        best_generated_result is None
        or best_generated_result["METEOR"] < metrics["METEOR"]
    ):
      best_generated_result = deepcopy(metrics)
  best_generated_result["perplexity_of_first_sentence"] \
      = float(perplexity_of_first_sentence)
  best_generated_result["pmid"] = abstract["pmid"]
  best_generated_result["title"] = title_only["text_data"][0]["text"]
  best_generated_result["mesh_headings"] = title_only["mesh_headings"]
  best_generated_result["date"] = title_only["date"]
  best_generated_result["references"] = reference_sentences
  return best_generated_result

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

    # Remember, we're using logsoftmax as output
    word_probabilities = np.exp(
        predictions["text"][-1, 0, :].detach().cpu().numpy()
    )

    assert 0 <= word_probabilities.min() <= word_probabilities.max() <= 1

    # Because floating point error may have slightly altered the probability dist

    choices = []
    probs = []
    for idx, prob in enumerate(word_probabilities):
      if prob > 0.001:
        choices.append(idx)
        probs.append(prob)

    probs = np.array(probs, dtype=np.float32)
    probs /= probs.sum()

    new_word = int(np.random.choice(choices, p=probs)) + tokenizer.vocab_start_idx
    yield new_word

    def add_or_shift(tensor, new_element):
      l = tensor.flatten().tolist()
      l.append(new_element)
      if len(l) >= model.max_text_length:
        l = l[-model.max_text_length+1:]
      return torch.LongTensor(l).unsqueeze(1).to(tensor.device)

    text = add_or_shift(text, new_word)

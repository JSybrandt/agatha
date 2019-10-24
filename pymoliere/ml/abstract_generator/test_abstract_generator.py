import pymoliere.ml.abstract_generator.util as util
from transformers import BertTokenizer, AdamW
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from copy import copy

SENTENCE_1 = (
    "Hypothesis generation is becoming a crucial time-saving technique which "
    "allows biomedical researchers to quickly discover implicit connections "
    "between important concepts."
)

SENTENCE_2 = (
    "Typically, these systems operate on domain-specific fractions of public "
    "medical data."
)

SENTENCE_3 = (
    "MOLIERE, in contrast, utilizes information from over 24.5 million "
    "documents."
)



def test_bert_generator_forward():
  """
  This test ensures that the modified version of bert can successfully do a
  single forward pass given two sentences.  The model should convert the hidden
  layers following the model to per-embedding logits.
  """
  model = util.AbstractGenerator.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  batch = pad_sequence(
      sequences=[
        torch.tensor(
          tokenizer.encode(
            SENTENCE_1,
            text_pair=SENTENCE_2,
            add_special_tokens=True
          )
        ),
        torch.tensor(
          tokenizer.encode(
            SENTENCE_2,
            text_pair=SENTENCE_3,
            add_special_tokens=True
          )
        ),
      ],
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  # batch of x, many tokens, softmax over voccab
  expected_shape = (batch.shape[0], batch.shape[1], tokenizer.vocab_size)
  res = model(batch)
  actual_shape = res.shape

  # Make sure that the above doesn't break
  assert expected_shape == actual_shape

def test_bert_generator_backwards():
  model = util.AbstractGenerator.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  loss_fn = torch.nn.NLLLoss()
  optim = AdamW(model.parameters(), lr=1)
  batch = pad_sequence(
      sequences=[
        torch.tensor(
          tokenizer.encode(
            SENTENCE_1,
            text_pair=SENTENCE_2,
            add_special_tokens=True
          )
        ),
        torch.tensor(
          tokenizer.encode(
            SENTENCE_2,
            text_pair=SENTENCE_3,
            add_special_tokens=True
          )
        ),
      ],
      batch_first=True,
      padding_value=tokenizer.pad_token_id,
  )
  # We're going to test that the last layer updates
  assert model.last_hidden2voccab.weight.requires_grad == True
  init_hidden_values = model.last_hidden2voccab.weight.detach().numpy().copy()

  # Super simple update model
  predicted_logits = model(batch)
  optim.zero_grad()
  # Loss expects a vec of nll vectors, and a vec of ints
  loss = loss_fn(
      predicted_logits.view(-1, tokenizer.vocab_size),
      batch.view(-1),
  )
  loss.backward()
  optim.step()

  final_hidden_values = model.last_hidden2voccab.weight.detach().numpy()
  print(init_hidden_values.shape)
  # We want to make sure that the weights have changed
  assert not np.allclose(init_hidden_values, final_hidden_values)


def test_apply_mask():
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  original_sequence = tokenizer.encode(
      text=SENTENCE_1,
      text_pair=SENTENCE_2,
      add_special_tokens=True
  )
  mask = [False] * len(original_sequence)
  mask[2] = mask[-2] = True
  # do the expected manually
  expected = copy(original_sequence)
  expected[-2] = expected[2] = tokenizer.mask_token_id
  expected[2] = tokenizer.mask_token_id
  # get actual
  actual = util.apply_mask_to_token_ids(
      tokenizer=tokenizer,
      input_ids=original_sequence,
      mask=mask
  )
  # require that actual is a modified copy
  assert original_sequence != actual
  assert actual == expected


def test_generate_sentence_mask():
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model_inputs = tokenizer.encode_plus(
      text=SENTENCE_1,
      text_pair=SENTENCE_2,
      add_special_tokens=True
  )
  input_ids = model_inputs['input_ids']
  segment_mask = model_inputs['token_type_ids']
  valid_mask_values = set([
    idx for idx, val in enumerate(segment_mask)
    if val == 1 and idx < len(segment_mask)-1
  ])
  # MASK EVERYTHING
  mask = util.generate_sentence_mask(
      segment_mask=segment_mask,
      per_token_mask_prob=1,
  )
  assert len(mask) == len(input_ids)
  # only marks those listed as valid
  for idx, is_masked in enumerate(mask):
    if is_masked:
      assert idx in valid_mask_values
  # marks all of the valid
  for idx in valid_mask_values:
    assert mask[idx]

def test_generate_sentence_mask_random():
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model_inputs = tokenizer.encode_plus(
      text=SENTENCE_1,
      text_pair=SENTENCE_2,
      add_special_tokens=True
  )
  input_ids = model_inputs['input_ids']
  segment_mask = model_inputs['token_type_ids']
  valid_mask_values = set([
    idx for idx, val in enumerate(segment_mask)
    if val == 1 and idx < len(segment_mask)-1
  ])
  # only marks those listed as valid
  expected_mask_prob = 0.25
  avg_marked_per_trial = []
  # run 1000 trials, we want to see if the result is close to 0.5
  for _ in range(1000):
    mask = util.generate_sentence_mask(
        segment_mask=segment_mask,
        per_token_mask_prob=expected_mask_prob,
    )
    assert len(mask) == len(input_ids)
    num_marked = 0
    for idx, is_masked in enumerate(mask):
      if is_masked:
        assert idx in valid_mask_values
        num_marked += 1
    avg_marked_per_trial.append(num_marked/len(valid_mask_values))
  assert np.isclose(
      np.mean(avg_marked_per_trial),
      expected_mask_prob,
      atol=0.02
  )

def test_group_sentences_into_pairs_sorted():
  records = [{
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_unsorted():
  # This sort order is difficult because when we see "first" we have to
  # generate both pairs at that time
  records = [{
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_bad_abstract():
  records = [{
        "sent_text": "first",
        "sent_idx": 1,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "title",
        "sent_idx": 0,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "ignore me!",
        "sent_idx": 0,
        "sent_total": 1,
        "pmid": 987,
        "version": 1,
      }, {
        "sent_text": "second",
        "sent_idx": 2,
        "sent_total": 3,
        "pmid": 123,
        "version": 1,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("title", "first"),
      ("first", "second"),
  ]
  assert set(actual) == set(expected)

def test_group_sentences_into_pairs_separate_versions():
  records = [{
        "sent_text": "v1 first",
        "sent_idx": 1,
        "sent_total": 2,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "v1 title",
        "sent_idx": 0,
        "sent_total": 2,
        "pmid": 123,
        "version": 1,
      }, {
        "sent_text": "v2 title",
        "sent_idx": 0,
        "sent_total": 2,
        "pmid": 123,
        "version": 2,
      }, {
        "sent_text": "v2 first",
        "sent_idx": 1,
        "sent_total": 2,
        "pmid": 123,
        "version": 2,
  }]
  actual = util.group_sentences_into_pairs(records)
  expected = [
      ("v1 title", "v1 first"),
      ("v2 title", "v2 first"),
  ]
  assert set(actual) == set(expected)

def test_sentence_pairs_to_model_io():
  sentence_pairs = [
      (SENTENCE_1, SENTENCE_2),
      (SENTENCE_2, SENTENCE_3),
  ]
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  model_kwargs, out_data = util.sentence_pairs_to_model_io(
      tokenizer=tokenizer,
      batch_pairs=sentence_pairs,
      unchanged_prob=0,
      full_mask_prob=0,
      mask_per_token_prob=0.8,
      max_sequence_length=500,
  )
  required_model_inputs = [
      'input_ids', 'attention_mask', 'token_type_ids',
  ]
  for val in required_model_inputs:
    assert val in model_kwargs
    assert model_kwargs[val].shape == out_data.shape


def test_generate_sentence():
  model = util.AbstractGenerator.from_pretrained("bert-base-uncased")
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

  next_sentence = util.generate_sentence(
      sentence=SENTENCE_1,
      model=model,
      tokenizer=tokenizer,
      max_sequence_length=500,
  )

  assert tokenizer.mask_token not in next_sentence
  assert tokenizer.unk_token not in next_sentence
  assert next_sentence != SENTENCE_1
  assert len(next_sentence.strip()) > 0


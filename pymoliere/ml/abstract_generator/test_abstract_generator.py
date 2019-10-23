import pymoliere.ml.abstract_generator.util as util
from transformers import BertTokenizer, AdamW
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

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

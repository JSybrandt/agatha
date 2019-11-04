#!/usr/bin/env python3
from pymoliere.ml.abstract_generator.util import AbstractGenerator, generate_sentence
from transformers import BertTokenizer, BertConfig
import torch
import sys

base_bert_version = "../data/scibert_scivocab_uncased"
config = BertConfig.from_pretrained(base_bert_version)
tokenizer = BertTokenizer.from_pretrained(base_bert_version)
model = AbstractGenerator(config)

pretrained_model = "/scratch4/jsybran/pymoliere_scratch/models/abstract_generator/model_gen4_2019_11_4.pt"
model.load_state_dict(torch.load(pretrained_model))
model.eval()



print("Play with the model!")
for sentence in sys.stdin:
  sentence = sentence.strip()
  for _ in range(4):
    sentence = generate_sentence(
        sentence=sentence,
        max_sequence_length=500,
        model=model,
        tokenizer=tokenizer,
    )
    print(sentence)

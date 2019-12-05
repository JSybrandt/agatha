#!/usr/bin/env python3
from fire import Fire
import json
from csv import DictWriter
from pathlib import Path
from typing import List

DEFAULT_FIELDS = [
    "pmid",
    "date",
    "mesh_headings",
    "title",
    "perplexity_of_first_sentence",
    "generated_text",
    "Bleu_1",
    "Bleu_2",
    "Bleu_3",
    "Bleu_4",
    "METEOR",
    "ROUGE_L",
    # Skipping CIDEr
    "SkipThoughtCS",
    "EmbeddingAverageCosineSimilairty",
    "VectorExtremaCosineSimilarity",
    "GreedyMatchingScore",
]

def to_csv(
  json_in:Path,
  csv_out:Path,
  fields:List[str]=DEFAULT_FIELDS
)->None:
  json_in = Path(json_in)
  csv_out = Path(csv_out)
  assert json_in.is_file()
  assert not csv_out.exists() and csv_out.parent.is_dir()

  with \
      open(json_in) as json_file, \
      open(csv_out, 'w', newline="") as csv_file:
    writer = DictWriter(csv_file, fields)
    writer.writeheader()
    for line in json_file:
      data = json.loads(line)
      data = {k: v for k, v in data.items() if k in DEFAULT_FIELDS}
      writer.writerow(data)


if __name__ == "__main__":
  Fire(to_csv)

#!/usr/bin/env python3

"""
This script converts the raw textual data provided in raw_data into the
expected input of the sentence classifier model. This entails embedding each
record, and converting the results into a mocked checkpoint directory.
"""

from argparse import ArgumentParser
from pathlib import Path
from pymoliere.ml.sentence_classifier import (
    util as sent_class_util,
    LABEL2IDX,
)
from pymoliere.construct import embedding_util
from pymoliere.util.misc_util import Record
from typing import Iterable
from pymoliere.construct import dask_process_global as dpg
import pickle


def parse_raw_file(raw_file_path:Path)->Iterable[Record]:
  docs = []
  with open(raw_file_path) as f:
    for line in f:
      line = line.strip()
      if len(line) == 0:
        continue
      if line[0] == "#":  # abstract header indicator
        docs.append([])
        continue
      docs[-1].append(line)

  res = []
  for doc in docs:
    for idx, line in enumerate(doc):
      try:
        label, text = line.split("\t", 1)
        label = f"abstract:{label.lower()}"
        assert label in LABEL2IDX
        res.append({
          "text": text,
          "sent_type": label,
          "sent_idx": (idx+1),
          "sent_total": len(doc),
          # It is important to select an invalid date that is larger than any
          # other because some classifier might be expecting date-based
          # training/validation/testing splits. However, because there is not
          # really a date associated with this record, any function thats
          # expecting a properly formatted YYYY-MM-DD date will fail, as
          # opposed to any function that is simply sorting strings.
          "date": "9999-99-99",
        })
      except:
        print(f"Err: '{line}'")
  return res


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--bert_model", type=str)
  parser.add_argument("--in_data_dir", type=Path)
  parser.add_argument("--out_data_dir", type=Path)
  parser.add_argument("--disable_gpu", action="store_true")
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--max_sequence_length", type=int, default=500)
  args = parser.parse_args()


  # Prepare and check input data
  in_train = args.in_data_dir.joinpath("train.txt")
  in_validation = args.in_data_dir.joinpath("dev.txt")
  in_test = args.in_data_dir.joinpath("test.txt")
  assert in_train.is_file()
  assert in_validation.is_file()
  assert in_test.is_file()

  # Prepare output data
  out_train = args.out_data_dir.joinpath("training_data")
  out_validation = args.out_data_dir.joinpath("validation_data")
  out_test = args.out_data_dir.joinpath("testing_data")
  out_train.mkdir(parents=True, exist_ok=True)
  out_validation.mkdir(parents=True, exist_ok=True)
  out_test.mkdir(parents=True, exist_ok=True)

  print("Setting up mock all_data")
  # The sentence classifier code is expecting an all_data checkpoint.
  # We don't actually use this if we've already got the train/validation/test
  # split
  out_all = args.out_data_dir.joinpath("all_data")
  out_all.mkdir(parents=True, exist_ok=True)
  out_all.joinpath("__done__").touch()

  print("Prepping embedding")
  preloader = dpg.WorkerPreloader()
  preloader.register(*embedding_util.get_pytorch_device_initalizer(
      disable_gpu=args.disable_gpu,
  ))
  preloader.register(*embedding_util.get_bert_initializer(
      bert_model=args.bert_model,
  ))
  dpg.add_global_preloader(preloader=preloader)

  for in_file, out_dir in [
      (in_train, out_train),
      (in_validation, out_validation),
      (in_test, out_test),
  ]:
    print(f"Converting {in_file} to {out_dir}")

    print("Converting to records.")
    records = parse_raw_file(in_file)

    # Step 3: Embed Records
    print("Embedding")
    embedded_records = embedding_util.embed_records(
        records,
        batch_size=args.batch_size,
        text_field="text",
        max_sequence_length=args.max_sequence_length,
        show_pbar=True,
    )

    # Step 4: Records to training data via util
    print("Converting to sentence_classifier.util.TrainingData")
    embedded_tuples = [
        sent_class_util.record_to_training_tuple(r)
        for r in embedded_records
    ]

    print("Saving as mock ckpt")
    done_file = out_dir.joinpath("__done__")
    part_file = out_dir.joinpath("part-0.pkl")
    with open(part_file, 'wb') as f:
      pickle.dump(embedded_tuples, f)
    with open(done_file, 'w') as f:
      f.write(f"{part_file.absolute()}\n")

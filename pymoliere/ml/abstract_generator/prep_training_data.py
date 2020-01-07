import dask
import dask.bag as dbag
from dask.distributed import Client
from pathlib import Path
import pickle
from pymoliere.config import config_pb2 as cpb
from pymoliere.construct import dask_checkpoint, file_util, text_util, ftp_util
from pymoliere.ml.abstract_generator.misc_util import OrderedIndex, items_to_ordered_index
from pymoliere.ml.abstract_generator.path_util import get_paths
from pymoliere.ml.abstract_generator import predicate_util
from pymoliere.util.misc_util import Record
import random
import sentencepiece as spm
from sqlitedict import SqliteDict
import string
from typing import Iterable, List, Dict, Optional
from pymoliere.construct import dask_process_global as dpg

def extract_predicates(config:cpb.AbstractGeneratorConfig):
  paths = get_paths(config)
  dask_client = connect_to_dask_cluster(config)

  preloader = dpg.WorkerPreloader()
  preloader.register(*predicate_util.get_scispacy_initalizer(
      config.predicate_spacy_model
  ))
  preloader.register(*predicate_util.get_stopwordlist_initializer(
      config.predicate_stopword_list
  ))
  dpg.add_global_preloader(client=dask_client, preloader=preloader)

  abstracts = file_util.load(
      paths["checkpoint_dir"]
      .joinpath("medline_documents")
  )

  predicates = abstracts.map_partitions(predicate_util.abstracts_to_predicates)
  predicates = dask_checkpoint.checkpoint(
      predicates,
      name="predicates",
      checkpoint_dir=paths["model_ckpt_dir"],
      overwrite=False,
  )
  predicates.compute()


def prep(config:cpb.AbstractGeneratorConfig):
  # all important paths
  paths = get_paths(config)
  connect_to_dask_cluster(config)
  def ckpt(val, name, overwrite=False):
    print("Checkpoint", name)
    return dask_checkpoint.checkpoint(
        val,
        name=name,
        checkpoint_dir=paths["model_ckpt_dir"],
        overwrite=overwrite,
    )

  # Get the full set of abstracts
  parsed_abstracts = (
      file_util.load(
        paths["checkpoint_dir"]
        .joinpath("sentences_with_lemmas")
      )
      .map_partitions(group_and_filter_parsed_sentences)
  )
  parsed_abstracts = ckpt(parsed_abstracts, "parsed_abstracts")

  is_test_data = (
      parsed_abstracts
      .map(lambda rec: (random.random() <= config.sys.test_ratio, rec))
  )
  is_test_data = ckpt(is_test_data, "is_test_data")

  testing_data = (
      is_test_data
      .filter(lambda b_r: b_r[0])
      .map(lambda b_r: b_r[1])
  )
  testing_data = ckpt(testing_data, "testing_data")

  training_data = (
      is_test_data
      .filter(lambda b_r: not b_r[0])
      .map(lambda b_r: b_r[1])
  )
  training_data = ckpt(training_data, "training_data")

  # write each partition of the training dataset to its own sqlitedict db
  # This allows for fast random access during distributed training
  print("Loading training database")
  to_training_database(training_data, paths["training_db_dir"])

  # print("Collecting all mesh headings")
  all_mesh_headings = (
      training_data
      .map(lambda rec: rec["mesh_headings"])
      .flatten()
      .frequencies()
      .filter(lambda mesh_freq: mesh_freq[1] >= config.min_mesh_term_support)
      .map(lambda mesh_freq: mesh_freq[0])
      .compute()
  )
  print(f"Indexing all {len(all_mesh_headings)} mesh headings")
  mesh_index = items_to_ordered_index(all_mesh_headings)

  ###

  print("Getting oldest year")
  oldest_year = (
      training_data
      .map(lambda rec: rec["year"])
      .filter(lambda year: year > 1000)  # some invalid years are crazy
      .min()
      .compute()
  )
  print("\t-", oldest_year)

  ###

  print("Collecting training data for tokenizer")
  training_data_files = (
      training_data
      # Only collect 30% of abstracts
      .random_sample(0.3)
      .map(lambda rec: [s["text"] for s in rec["sentences"]])
      .flatten()
      # Only take 10% of sentences, ultimately,'re subsetting again
      .random_sample(0.1)
      .map(lambda text: text.lower() if config.lowercase else text)
      # Reduce the total number of files
      .repartition(20)
      # Store results in textfiles
      .to_textfiles(f"{paths['tokenizer_training_data_dir']}/*.txt")
  )
  print("Training tokenizer")
  # need to place files in tokenizer_model_path
  spm.SentencePieceTrainer.train(
      f"--input={','.join(training_data_files)} "
      f"--model_prefix={paths['tokenizer_model_path'].parent}/tokenizer "
      f"--vocab_size={config.vocab_size} "
      f"--character_coverage=1.0 "
      f"--model_type=unigram "
      f"--input_sentence_size={config.max_tokenizer_sentences} "
      f"--shuffle_input_sentence=true "
  )
  assert paths["tokenizer_model_path"].is_file()
  assert paths["tokenizer_vocab_path"].is_file()

  extra_data = {
      "mesh_index": mesh_index,
      "oldest_year": oldest_year,
  }
  with open(paths["model_extra_data_path"], 'wb') as f:
    pickle.dump(extra_data, f)
  print("\t- Written:", paths["model_extra_data_path"])


def group_and_filter_parsed_sentences(
    sentences:Iterable[Record]
)->Iterable[Record]:
  # use pmid:version to index
  abstracts = {}
  for sentence in sentences:
    # The abstract must contain at least three sentences. This discards
    # many 1-sentence title-only abstracts
    if sentence["sent_total"] >= 3:
      key = f"{sentence['pmid']}:{sentence['version']}"
      if key not in abstracts:
        abstracts[key] = {
            "pmid": sentence["pmid"],
            "year": int(sentence["date"].split("-")[0]),  # YYYY-MM-DD
            "mesh_headings": sentence["mesh_headings"][:],
            "sentences": [None] * sentence["sent_total"],
        }
      abstracts[key]["sentences"][sentence["sent_idx"]] = {
          "text": sentence["sent_text"],
          "type": sentence["sent_type"],
          "tags": [
            (t["cha_start"], t["cha_end"], t["pos"], t["dep"])
            for t in sentence["tokens"]
          ],
          "ents": [
            (e["cha_start"], e["cha_end"], e["label"])
            for e in sentence["entities"]
          ],
      }
  return list(abstracts.values())

def connect_to_dask_cluster(
    config:cpb.AbstractGeneratorConfig
)->Optional[Client]:
  # Potential cluster
  if config.cluster.run_locally or config.cluster.address == "localhost":
    print("Running dask on local machine!")
    return None
  else:
    cluster_address = f"{config.cluster.address}:{config.cluster.port}"
    print("Configuring Dask, attaching to cluster")
    print(f"\t- {cluster_address}")
    dask_client = Client(address=cluster_address)
    if config.cluster.restart:
      print("\t- Restarting cluster...")
      dask_client.restart()
    return dask_client


def to_training_database(bag:dbag.Bag, database_dir:Path):
  assert database_dir.is_dir()
  done_file = database_dir.joinpath("__done__")
  if not done_file.is_file():
    def part_to_db(records):
      r_name = "".join([random.choice(string.ascii_letters) for _ in range(10)])
      db_path = database_dir.joinpath(r_name + ".sqlite")
      with SqliteDict(db_path, journal_mode="OFF", flag="n") as db:
        for idx, rec in enumerate(records):
          db[str(idx)] = rec
        db.commit()
      return db_path
    db_paths = bag.map_partitions(part_to_db).compute()
    with open(done_file, 'w') as f:
      for p in db_paths:
        f.write(f"{p}\n")

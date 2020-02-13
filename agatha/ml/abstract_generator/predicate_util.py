from collections import defaultdict
from itertools import product
from predpatt import PredPatt, load_conllu
import re
import spacy
from spacy_conll import Spacy2ConllParser
from pathlib import Path
from typing import Iterable, Tuple
from agatha.construct import dask_process_global as dpg
from agatha.util.misc_util import Record

def get_scispacy_initalizer(
    scispacy_version:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    return spacy.load(scispacy_version)
  return "pred_util:nlp", _init

def get_stopwordlist_initializer(
    stopword_path:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    with open(stopword_path, 'r') as f:
      return {line.strip().lower() for line in f}
  return "pred_util:stopwords", _init

def generate_predicates(
    abstract_text:str,
    pred_patt_opts=None
)->Iterable[Tuple[str, str, str]]:
  "Requires that pred_util:nlp and pred_util:stopwords be initialized"
  nlp = dpg.get("pred_util:nlp")
  parser = Spacy2ConllParser(nlp=nlp)
  stopwords = dpg.get("pred_util:stopwords")

  doc = nlp(abstract_text)
  for sent in doc.sents:
    # if the sentence is very long
    if len(sent) >= 20:
      word_count = defaultdict(int)
      for tok in sent:
        word_count[str(tok)] += 1
        # if one word dominates the long sentence
      if max(word_count.values()) >= len(sent)*0.2:
        continue  # we likely generated the same word over-and-over
    conllu = "".join(list(parser.parse(input_str=str(sent))))
    for _, pred_patt_parse in load_conllu(conllu):
      predicates = PredPatt(
        pred_patt_parse,
        opts=pred_patt_opts
      ).instances
      for predicate in predicates:
        # We only care about 2-entity predicates
        if len(predicate.arguments) == 2:
          a_ents, b_ents = [
              # Get the set of entities
              filter(
                # Not in the stopword list
                lambda x: x not in stopwords,
                [str(e).strip() for e in nlp(args.phrase()).ents]
              )
              # For each argument
              for args in predicate.arguments
          ]
          # Slight cleaning needed to better match the predicate phrase
          # Note, that PredPatt predicates use ?a and ?b placeholders
          predicate_stmt = (
              re.match(
                r".*\?a(.*)\?b.*", # get text between placeholders
                predicate.phrase()
              )
              .group(1) # get the group matched between the placeholders
              .strip()
          )
          if len(predicate_stmt) > 0:
            # We're going to iterate all predicates
            for a, b in product(a_ents, b_ents):
              if a != b:
                yield (a, predicate_stmt, b)

def abstracts_to_predicates(abstracts:Iterable[Record])->Iterable[Record]:
  res = []
  for record in abstracts:
    text = " ".join([
      td["text"] for td in record["text_data"] if td["type"] != "title"
    ])
    for subj, predicate_stmt, obj in generate_predicates(text):
      res.append(dict(
        pmid=record["pmid"],
        date=record["date"],
        subj=subj,
        obj=obj,
        predicate_stmt=predicate_stmt,
      ))
  return res

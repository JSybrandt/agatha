from copy import copy
from nltk.tokenize import sent_tokenize
from pathlib import Path
from pymoliere.construct import dask_process_global as dpg
from pymoliere.util import db_key_util, misc_util
from pymoliere.util.misc_util import Record, Edge
from typing import List, Tuple, Any, Optional, Dict, Callable, Iterable, Set
import dask.bag as dbag
import logging
import math
import spacy
from dask import delayed

# More details : https://spacy.io/api/annotation
INTERESTING_POS_TAGS = {
    "NOUN",
    "VERB",
    "ADJ",
    "PROPN",  # proper noun
    "ADV", # adverb
    "INTJ", # interjection
    "X", # other
}

#####################

def _op_g_pre(graph:bool)->str:
  "helper for graph ids"
  return f"{db_key_util.GRAPH_TYPE}:" if graph else ""

def sentence_to_id(sent:Record, graph:bool=False)->str:
  return get_sentence_id(
      pmid=sent["pmid"],
      version=sent["version"],
      sent_idx=sent["sent_idx"],
      graph=graph,
  )

def get_sentence_id(
    pmid:int,
    version:int,
    sent_idx:int,
    graph:bool=False
)->str:
  typ = db_key_util.SENTENCE_TYPE
  return f"{_op_g_pre(graph)}{typ}:{pmid}:{version}:{sent_idx}"

def token_to_id(
    token:Record,
    graph:bool=False,
)->str:
  typ = db_key_util.LEMMA_TYPE
  lem = token["lemma"]
  pos = token["pos"]
  return f"{_op_g_pre(graph)}{typ}:{pos}:{lem}".lower()

def entity_to_id(
    entity:Record,
    sentence:Record,
    token_field:str="tokens",
    graph:bool=False,
)->str:
  ent = get_entity_text(
      entity=entity,
      sentence=sentence,
      token_field=token_field
  )
  typ = db_key_util.ENTITY_TYPE
  return f"{_op_g_pre(graph)}{typ}:{ent}".lower()

def get_entity_text(
    entity:Record,
    sentence:Record,
    token_field:str="tokens",
)->str:
  return "_".join([
      sentence[token_field][tok_idx]["lemma"]
      for tok_idx in range(entity["tok_start"], entity["tok_end"])
  ])

def mesh_to_id(
    mesh_code:str,
    graph:bool=False,
)->str:
  typ = db_key_util.MESH_TERM_TYPE
  return f"{_op_g_pre(graph)}{typ}:{mesh_code}".lower()

def ngram_to_id(
    ngram_text:str,
    graph:bool=False,
)->str:
  typ = db_key_util.NGRAM_TYPE
  return f"{_op_g_pre(graph)}{typ}:{ngram_text}".lower()

################################################################################

def get_scispacy_initalizer(
    scispacy_version:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    return spacy.load(scispacy_version)
  return "text_util:nlp", _init

def get_stopwordlist_initializer(
    stopword_path:Path
)->Tuple[str, dpg.Initializer]:
  def _init():
    with open(stopword_path, 'r') as f:
      return {line.strip().lower() for line in f}
  return "text_util:stopwords", _init

################################################################################

def split_sentences(
    document_elem:Record,
    text_data_field:str="text_data",
    id_field:str="id",
    min_sentence_len:Optional[int]=None,
    max_sentence_len:Optional[int]=None,
)->List[Record]:
  """
  Splits a document into its collection of sentences. In order of text field
  elements, we split sentences and create new elements for the result. All
  fields from the original document, as well as the text field (minus the
  actual text itself) are copied over.

  If min/max sentence len are specified, we do NOT consider sentences that fail
  to match the range.

  `id_field` will be set with {SENTENCE_TYPE}:{pmid}:{version}:{sent_idx}

  For instance:

  {
    "status": "Published",
    "umls": ["C123", "C456"],
    "text_fields": [{
        "text": "Title 1",
        "type": "title"
      }, {
        "text": "This is an abstract. This is another sentence.",
        "type": "abstract:raw",
      }]
  }

  becomes:

  [{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "Title 1",
    "sent_type": "title",
    "sent_idx": 0,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is an abstract.",
    "sent_type": "abstract:raw",
    "sent_idx": 1,
    "sent_total": 3,
    },{
    "status": "Published",
    "umls": ["C123", "C456"],
    "sent_text": "This is another sentence.",
    "sent_type": "abstract:raw",
    "sent_idx": 2,
    "sent_total": 3,
  }]
  """
  assert text_data_field in document_elem
  sent_text_key = f"sent_text"
  sent_idx_key = f"sent_idx"
  sent_total_key = f"sent_total"

  res = []

  # Get all non-textual data
  doc_non_text_elem = copy(document_elem)
  doc_non_text_elem.pop(text_data_field)

  sent_idx = 0
  for text_data in document_elem[text_data_field]:
    assert "text" in text_data
    # Holds all additional text features per-sentence in this field
    non_text_attr = copy(doc_non_text_elem)
    for key, val in text_data.items():
      if key != "text":
        key = f"sent_{key}"
        assert key not in non_text_attr
        non_text_attr[key] = val
    for sentence_text in sent_tokenize(text_data["text"]):
      if min_sentence_len is not None and len(sentence_text) < min_sentence_len:
        continue
      if max_sentence_len is not None and len(sentence_text) > max_sentence_len:
        continue
      sent_rec = copy(non_text_attr)
      assert sent_text_key not in sent_rec
      sent_rec[sent_text_key] = sentence_text
      assert sent_idx_key not in sent_rec
      sent_rec[sent_idx_key] = sent_idx
      sent_idx += 1
      res.append(sent_rec)

  # set total
  for r in res:
    r[sent_total_key] = sent_idx
    r[id_field] = sentence_to_id(r)
  return res

def get_frequent_ngrams(
    analyzed_sentences:dbag.Bag,
    max_ngram_length:int,
    min_ngram_support:int,
    min_ngram_support_per_partition:int,
    token_field:str="tokens",
    ngram_field:str="ngrams"
)->dbag.Bag:
  """
  Adds a new field containing a list of all mined n-grams.  N-grams are tuples
  of strings such that at least one string is not a stopword.  Strings are
  collected from the lemmas of sentences.  To be counted, an ngram must occur
  in at least `min_ngram_support` sentences.
  """
  def part_to_ngram_counts(
      records:Iterable[Record]
  )->Iterable[Dict[Tuple[str], int]]:
    ngram2count = {}
    for rec in records:
      for start_tok_idx in range(len(rec[token_field])):
        for ngram_len in range(2, max_ngram_length):
          end_tok_idx = start_tok_idx + ngram_len
          if end_tok_idx > len(rec[token_field]):
            continue
          relevant_words = 0
          for tok_idx in range(start_tok_idx, end_tok_idx):
            if not rec[token_field][tok_idx]["stop"] and \
                rec[token_field][tok_idx]["pos"] in INTERESTING_POS_TAGS:
              relevant_words += 1
          if relevant_words >= 2:
            ngram = tuple(
                rec[token_field][tok_idx]["lemma"]
                for tok_idx
                in range(start_tok_idx, end_tok_idx)
            )
            if ngram in ngram2count:
              ngram2count[ngram] += 1
            else:
              ngram2count[ngram] = 1
    # filter out all low-occurrence ngrams in this partition
    return [{
        n: c for n, c in ngram2count.items()
        if c >= min_ngram_support_per_partition
    }]

  def valid_ngrams(ngram2count:Dict[str,int])->Set[Tuple[str]]:
    ngrams =  {
        n for n, c in ngram2count.items()
        if c >= min_ngram_support
    }
    return ngrams

  def parse_ngrams(record:Record, ngram_model:Set[Tuple[str]]):
    record[ngram_field] = []
    start_tok_idx = 0
    while start_tok_idx < len(record[token_field]):
      incr = 1  # amount to move start_tok_idx
      # from max -> 2. Match longest
      for ngram_len in range(max_ngram_length, 1, -1):
        # get bounds of ngram and make sure its within sentence
        end_tok_idx = start_tok_idx + ngram_len
        if end_tok_idx > len(record[token_field]):
          continue
        ngram = tuple(
            record[token_field][tok_idx]["lemma"]
            for tok_idx in range(start_tok_idx, end_tok_idx)
        )
        # if match
        if ngram in ngram_model:
          record[ngram_field].append("_".join(ngram))
          # skip over matched terms
          incr = ngram_len
          break
      start_tok_idx += incr
    return record

  # Begin the actual function
  if max_ngram_length < 1:
    # disable, record empty field for all ngrams
    def init_nothing(rec:Record)->Record:
      rec[ngram_field]=[]
      return rec
    return analyzed_sentences.map(init_nothing)
  else:
    ngram2count = analyzed_sentences.map_partitions(
        part_to_ngram_counts
    ).fold(
        misc_util.merge_counts,
        initial={}
    )
    ngram_model = delayed(valid_ngrams)(ngram2count)
    return analyzed_sentences.map(parse_ngrams, ngram_model=ngram_model)


def analyze_sentences(
    records:Iterable[Record],
    text_field:str,
    token_field:str="tokens",
    entity_field:str="entities",
)->Iterable[Record]:
  """
  Parses the text fields of all records using SciSpacy.
  Requires that text_util:nlp and text_util:stopwords have both been loaded into
  dask_process_global.

  @param records: A partition of records to parse, each must contain `text_field`
  @param text_field: The name of the field we wish to parse.
  @param token_field: The output field for all basic tokens. These are
  sub-records containing information such as POS tag and lemma.
  @param entity_field: The output field for all entities, which are multi-token
  phrases.
  @return a list of records with token and entity fields
  """
  nlp = dpg.get("text_util:nlp")
  stopwords = dpg.get("text_util:stopwords")

  res = []
  for sent_rec, doc in zip(
      records,
      nlp.pipe(
        map(
          lambda x:x[text_field],
          records
        ),
      batch_size=10000)
  ):
    sent_rec[entity_field] = [
      {
        "tok_start": ent.start,
        "tok_end": ent.end,
        "cha_start": ent.start_char,
        "cha_end": ent.end_char,
      }
      for ent in doc.ents
      if ent.end - ent.start > 1  # don't want 1-gram ents
    ]
    sent_rec[token_field] = [
        {
          "cha_start": tok.idx,
          "cha_end": tok.idx + len(tok),
          "lemma": tok.lemma_,
          "pos": tok.pos_,
          "tag": tok.tag_,
          "dep": tok.dep_,
          "stop": \
              tok.lemma_ in stopwords or tok.text.strip().lower() in stopwords
        }
        for tok in doc
    ]
    res.append(sent_rec)
  return res


def add_bow_to_analyzed_sentence(
    record:Record,
    bow_field="bow",
    token_field="tokens",
    entity_field="entities",
    mesh_heading_field="mesh_headings",
    ngram_field="ngrams",
)->Record:
  bow = []
  for lemma in record[token_field]:
    if lemma["pos"] in INTERESTING_POS_TAGS and not lemma["stop"]:
      bow.append(lemma["lemma"])
  for entity in record[entity_field]:
    is_stop_ent = True
    for t in record[token_field][entity["tok_start"]:entity["tok_end"]]:
      if not t["stop"]:
        is_stop_ent = False
        break
    # A stop-entity is one comprised of only stopwords such as "et_al."
    if not is_stop_ent:
      ent_text = get_entity_text(
          entity=entity,
          sentence=record,
          token_field=token_field
      )
      bow.append(ent_text)
  bow += record[ngram_field]
  bow += record[mesh_heading_field]
  record[bow_field] = bow
  return record

def get_document_frequencies(
    analyzed_sentences:dbag.Bag,
    token_field:str="tokens",
    entity_field:str="entities",
    mesh_field:str="mesh_headings",
    ngram_field:str="ngrams"
)->Record:  # delayed counts
  def record_to_init_counts(
      record:Record,
  )->Record:
    res = {}
    for field, to_key in [
        (token_field, lambda x:token_to_id(x, graph=True)),
        (entity_field, lambda x: entity_to_id(x,
                                              sentence=record,
                                              token_field=token_field,
                                              graph=True)
        ),
        (mesh_field, lambda x:mesh_to_id(x, graph=True)),
        (ngram_field, lambda x:ngram_to_id(x, graph=True)),
    ]:
      for val in record[field]:
        key = to_key(val)
        res[key] = 1
    return res
  return analyzed_sentences.map(
      record_to_init_counts
  ).fold(
      misc_util.merge_counts,
      initial={}
  )


def calc_tf_idf(
    terms:List[str],
    document_freqs:Dict[str, int],
    total_documents:int,
)->Dict[str, float]:
  tfs = {}
  for t in terms:
    if t in tfs:
      tfs[t] += 1
    else:
      tfs[t] = 1
  return {
      t: (
        # tf. Log of document length reduces variance of small docs
        f/(math.log(len(terms))+1)
      )*(
        # idf
        math.log(total_documents/document_freqs[t])
      )
      for t, f in tfs.items()
      if t in document_freqs
  }


def get_edges_from_sentence_part(
    sentences:Iterable[Record],
    document_freqs:Dict[str,int],
    total_documents:int,
    token_field:str="tokens",
    entity_field:str="entities",
    mesh_field:str="mesh_headings",
    sent_idx_field:str="sent_idx",
    sent_total_field:str="sent_total",
    ngram_field:str="ngrams",
    id_field:str="id",
)->Iterable[Edge]:
  res = []
  for sent_rec in sentences:
    sent_k = db_key_util.to_graph_key(sent_rec[id_field])
    # Calculate
    keys = []
    for token in sent_rec[token_field]:
      if not token["stop"] and token["pos"] in INTERESTING_POS_TAGS:
        tok_k = token_to_id(token, graph=True)
        keys.append(tok_k)
    for entity in sent_rec[entity_field]:
      ent_k = entity_to_id(
          entity=entity,
          sentence=sent_rec,
          token_field=token_field,
          graph=True
      )
      keys.append(ent_k)
    for mesh_code in sent_rec[mesh_field]:
      keys.append(mesh_to_id(mesh_code, graph=True))
    for ngram in sent_rec[ngram_field]:
      keys.append(ngram_to_id(ngram, graph=True))

    for tok_k, tf_idf in calc_tf_idf(
        terms=keys,
        document_freqs=document_freqs,
        total_documents=total_documents
    ).items():
      weight = 1/tf_idf
      res.append(db_key_util.to_edge(
        source=sent_k,
        target=tok_k,
        weight=weight
      ))
      res.append(db_key_util.to_edge(
        target=sent_k,
        source=tok_k,
        weight=weight
      ))
    # Adj sentence edges. We only need to make edges for "this" sentence,
    # because the other sentences will get the other sides of each connection.
    if sent_rec[sent_idx_field] > 0:
      res.append(
        db_key_util.to_edge(
          source=sent_k,
          target=get_sentence_id(
            pmid=sent_rec["pmid"],
            version=sent_rec["version"],
            sent_idx=sent_rec["sent_idx"]-1,
            graph=True,
          )
        )
      )
    if sent_rec[sent_idx_field] < sent_rec[sent_total_field]-1:
      res.append(
        db_key_util.to_edge(
          source=sent_k,
          target=get_sentence_id(
            pmid=sent_rec["pmid"],
            version=sent_rec["version"],
            sent_idx=sent_rec["sent_idx"]+1,
            graph=True,
          )
        )
      )
  return res

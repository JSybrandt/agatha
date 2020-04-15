import dask
import dask.bag as dbag
from typing import Iterable, Dict, Tuple, Set
from agatha.util.misc_util import Record, merge_counts
from agatha.construct.text_util import INTERESTING_POS_TAGS

def get_frequent_ngrams(
    analyzed_sentences:dbag.Bag,
    max_ngram_length:int,
    min_ngram_support:int,
    min_ngram_support_per_partition:int,
    ngram_sample_rate:float,
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

      def interesting(idx):
        t = rec[token_field][idx]
        return not t["stop"] and t["pos"] in INTERESTING_POS_TAGS

      # beginning of ngram
      for start_tok_idx in range(len(rec[token_field])):
        # ngrams must begin with an interesting word
        if not interesting(start_tok_idx):
          continue
        # for each potential n-gram size
        for ngram_len in range(2, max_ngram_length):
          end_tok_idx = start_tok_idx + ngram_len
          # ngrams cannot extend beyond the sentence
          if end_tok_idx > len(rec[token_field]):
            continue
          # ngrams must end with an interesting word
          if not interesting(end_tok_idx-1):
            continue
          # the ngram is an ordered tuple of lemmas
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
    ngram2count = (
        analyzed_sentences
        .random_sample(ngram_sample_rate)
        .map_partitions(part_to_ngram_counts)
        .fold(merge_counts, initial={})
    )
    ngram_model = dask.delayed(valid_ngrams)(ngram2count)
    return analyzed_sentences.map(parse_ngrams, ngram_model=ngram_model)

"""
This module is responsible for adding auxiliary helper data to the result proto
"""

from agatha.topic_query import (
    topic_query_result_pb2 as res_pb,
)
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np
from typing import Iterable, Tuple, Dict, Optional
from itertools import combinations
from agatha.util.sqlite3_lookup import Sqlite3Graph, Sqlite3Bow
from collections import defaultdict
from agatha.util.entity_types import is_sentence_type
import itertools


def _graph_bfs_iterator(start_key:str, graph_db:Sqlite3Graph)->Iterable[str]:
  """
  Iterates graph keys, starting with `start_key` and going outwards in a bfs
  manner
  """
  queue = [start_key]
  visited = {start_key}
  while len(queue) > 0:
    current_key = queue.pop()
    yield current_key
    for neigh in graph_db[current_key]:
      if neigh not in visited:
        queue.append(neigh)
        visited.add(neigh)

def estimate_plaintext_from_graph_key(
  graph_key:str,
  graph_db:Sqlite3Graph,
  bow_db:Sqlite3Bow,
  num_sent_to_check:int=100,
)->Optional[str]:
  """
  Given a graph key, get the most likely plaintext word associated with it.
  For instance, given "l:noun:cancer" or "m:d009369" we should get something
  like "cancer"
  """

  word2count = defaultdict(int)

  # Select a subset of sentences to lookup
  sentences_to_check = itertools.islice(
      filter(
        is_sentence_type,
        _graph_bfs_iterator(
          graph_key,
          graph_db
        )
      ),
      num_sent_to_check
  )

  for neighbor in sentences_to_check:
    if neighbor in bow_db:
      for word in bow_db[neighbor]:
        word2count[word] += 1

  max_count = None
  res = None
  for word, count in word2count.items():
    if max_count is None or count > max_count:
      max_count = count
      res = word
  return res


def _weighted_jacc(a:np.array, b:np.array)->float:
  return np.sum(np.minimum(a, b)) / np.sum(np.maximum(a, b))


def _all_pairs_jaccard_comparisions(
  graph_idx2vec:Dict[int, np.array]
)->Iterable[Tuple[int, int, float]]:
  """
  Performs all-pairs comparisons within the set of vectors.  If there is a
  nonzero similarity between vectors i, and j, then we will generate: (i, j,
  sim(i,j)), and (j, i, sim(i,j)).
  note: Similarity is symmetrical.
  """

  for i, j in combinations(graph_idx2vec.keys(), 2):
    sim = _weighted_jacc(graph_idx2vec[i], graph_idx2vec[j])
    if sim > 0:
      yield (i, j, sim)
      yield (j, i, sim)


def add_topical_network(
    result:res_pb.TopicQueryResult,
    topic_model:LdaModel,
    dictionary:Dictionary,
    graph_db:Sqlite3Graph,
    bow_db:Sqlite3Bow,
)->None:
  """
  Adds the topical_network field to the result proto.
  Creates this network by the weighted jacquard of topics.

  The source and target words are going to be assigned indices -1 and -2.
  """
  # Size n_topics X voccab_size
  term_topic_mat = topic_model.get_topics()
  num_topics, vocab_size = term_topic_mat.shape

  source_word = estimate_plaintext_from_graph_key(
      graph_key=result.source,
      graph_db=graph_db,
      bow_db=bow_db,
  )
  assert source_word is not None, \
      f"Failed to find plaintext entry for {result.source}"
  source_word_idx = dictionary.token2id[source_word]
  source_graph_idx = -1
  source_vec = np.zeros(vocab_size)
  source_vec[source_word_idx] = 1

  target_word = estimate_plaintext_from_graph_key(
      graph_key=result.target,
      graph_db=graph_db,
      bow_db=bow_db,
  )
  assert target_word is not None, \
      f"Failed to find plaintext entry for {result.target}"
  target_word_idx = dictionary.token2id[target_word]
  target_graph_idx = -2
  target_vec = np.zeros(vocab_size)
  target_vec[target_word_idx] = 1

  graph_idx2vec = {
      topic_idx: term_topic_mat[topic_idx, :]
      for topic_idx in range(num_topics)
  }
  graph_idx2vec[source_graph_idx] = source_vec
  graph_idx2vec[target_graph_idx] = target_vec

  # Set all node names
  for idx in range(num_topics):
    result.topical_network.nodes[idx].name = f"Topic: {idx}"
  result.topical_network.nodes[source_graph_idx].name = \
      f"Source: '{result.source}' -- '{source_word}'"
  result.topical_network.nodes[target_graph_idx].name = \
      f"Source: '{result.target}' -- '{target_word}'"

  # Set all edges:
  for i, j, sim in _all_pairs_jaccard_comparisions(graph_idx2vec):
    result.topical_network.nodes[i].neighbors[j] = sim

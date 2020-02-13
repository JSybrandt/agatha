from typing import List, Dict, Set
from agatha.util.misc_util import merge_counts

def get_document_frequencies(text_documents:List[List[str]])->Dict[str, int]:
  "Returns the document occurrence rate for words across documents"
  res = {}
  for doc in text_documents:
    res = merge_counts(res, {t: 1 for t in set(doc)})
  return res

def filter_words(
    keys:List[str],
    text_corpus:List[List[str]],
    stopwords:Set[str]
)->List[List[str]]:
  filtered_keys = []
  filtered_docs = []
  for key, doc in zip(keys, text_corpus):
    doc = list(filter(lambda x: x not in stopwords, doc))
    if len(doc) > 0:
      filtered_keys.append(key)
      filtered_docs.append(doc)
  return filtered_keys, filtered_docs

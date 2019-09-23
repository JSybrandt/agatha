from typing import List, Dict, Set
from pymoliere.util.misc_util import merge_counts
def get_document_frequencies(text_documents:List[List[str]])->Dict[str, int]:
  "Returns the document occurrence rate for words across documents"
  res = {}
  for doc in text_documents:
    res = merge_counts(res, {t: 1 for t in set(doc)})
  return res

def filter_words(
    text_documents:List[List[str]],
    stopwords:Set[str]
)->List[List[str]]:
  res = []
  for doc in text_documents:
    doc = list(filter(lambda x: x not in stopwords, doc))
    if len(doc) > 0:
      res.append(doc)
  return res


from pymoliere.query import bow_util

def test_filter_words():
  keys = [1, 2, 3]
  text_corpus = [
      ["a", "b", "c"],
      ["b"],
      ["b", "c", "d"],
  ]
  stopwords = {"b", "d"}
  expected_keys = [1, 3]
  expected_corpus = [
      ["a", "c"],
      ["c"],
  ]
  actual_keys, actual_corpus = bow_util.filter_words(
      keys=keys,
      text_corpus=text_corpus,
      stopwords=stopwords,
  )
  assert actual_keys==expected_keys
  assert actual_corpus==expected_corpus

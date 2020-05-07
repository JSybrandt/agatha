from agatha.ml.hypothesis_predictor import predicate_util

def test_clean_coded_term():
  term = "c0444567"
  expected = "m:c0444567"
  actual = predicate_util.clean_coded_term(term)
  assert actual == expected

def test_clean_coded_term_passthrough():
  term = "m:c0444567"
  expected = "m:c0444567"
  actual = predicate_util.clean_coded_term(term)
  assert actual == expected

def test_clean_coded_term_lower():
  term = "C0444567"
  expected = "m:c0444567"
  actual = predicate_util.clean_coded_term(term)
  assert actual == expected

def test_clean_coded_term_passthrough_lower():
  term = "m:C0444567"
  expected = "m:c0444567"
  actual = predicate_util.clean_coded_term(term)
  assert actual == expected

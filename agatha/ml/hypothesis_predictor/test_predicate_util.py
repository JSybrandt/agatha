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

def test_is_valid_predicate_name():
  assert predicate_util.is_valid_predicate_name("p:c123:CAUSES:c456")
  assert predicate_util.is_valid_predicate_name("p:123:CAUSES:456")
  assert predicate_util.is_valid_predicate_name("p:1:FOO:4")
  assert not predicate_util.is_valid_predicate_name("p:c123:CAUSES:")
  assert not predicate_util.is_valid_predicate_name("p::CAUSES:c456")
  assert not predicate_util.is_valid_predicate_name("c123:CAUSES:c456")
  assert not predicate_util.is_valid_predicate_name("p:")
  assert not predicate_util.is_valid_predicate_name("p:1:2")
  assert not predicate_util.is_valid_predicate_name("p:::")
  assert not predicate_util.is_valid_predicate_name("")

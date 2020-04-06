from agatha.construct.document_parsers import parse_covid_json, document_record
import json

from pathlib import Path

COVID_JSON_1 = Path("./test_data/example_covid_1.json")
assert COVID_JSON_1.is_file(), f"Failed to find testing data: {COVID_JSON_1}"

COVID_JSON_2 = Path("./test_data/example_covid_2.json")
assert COVID_JSON_2.is_file(), f"Failed to find testing data: {COVID_JSON_2}"

with open(COVID_JSON_1, 'rb') as json_file:
  covid_raw_1 = json.load(json_file)

with open(COVID_JSON_2, 'rb') as json_file:
  covid_raw_2 = json.load(json_file)

def get_expected_texts(covid_raw):
  res = []
  for field in ["abstract", "body_text"]:
    if field in covid_raw:
      for ent in covid_raw[field]:
        res.append(ent["text"])
  for _, ent in covid_raw["ref_entries"].items():
    res.append(ent["text"])
  return res

def test_parse_success_1():
  "Ensures that parsing happens without failure"
  record = parse_covid_json.json_path_to_record(COVID_JSON_1)
  document_record.assert_valid_document_record(record)

def test_parse_success_2():
  "Ensures that parsing happens without failure"
  record = parse_covid_json.json_path_to_record(COVID_JSON_2)
  document_record.assert_valid_document_record(record)

def test_parse_title_1():
  record = parse_covid_json.json_path_to_record(COVID_JSON_1)
  record["title"] = covid_raw_1["metadata"]["title"]

def test_parse_title_2():
  record = parse_covid_json.json_path_to_record(COVID_JSON_2)
  record["title"] = covid_raw_2["metadata"]["title"]

def test_parse_text_1():
  record = parse_covid_json.json_path_to_record(COVID_JSON_1)
  actual_texts = [td["text"] for td in record["text_data"]]
  expected_texts = get_expected_texts(covid_raw_1)
  assert set(actual_texts) == set(expected_texts)

def test_parse_text_2():
  record = parse_covid_json.json_path_to_record(COVID_JSON_2)
  actual_texts = [td["text"] for td in record["text_data"]]
  expected_texts = get_expected_texts(covid_raw_2)
  assert set(actual_texts) == set(expected_texts)

def test_parse_id_1():
  record = parse_covid_json.json_path_to_record(COVID_JSON_1)
  actual_id = record["pmid"]
  expected_id = covid_raw_1["paper_id"]
  assert actual_id == expected_id

def test_parse_id_2():
  record = parse_covid_json.json_path_to_record(COVID_JSON_2)
  actual_id = record["pmid"]
  expected_id = covid_raw_2["paper_id"]
  assert actual_id == expected_id

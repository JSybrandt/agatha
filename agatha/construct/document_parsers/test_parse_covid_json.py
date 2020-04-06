from agatha.construct.document_parsers import parse_covid_json, document_record
import json

from pathlib import Path

COVID_JSON = Path("./test_data/example_covid.json")
assert COVID_JSON.is_file(), f"Failed to find testing data: {COVID_JSON}"

with open(COVID_JSON, 'rb') as json_file:
  covid_raw_json = json.load(json_file)

def get_expected_texts():
  res = []
  for field in ["abstract", "body_text"]:
    for ent in covid_raw_json[field]:
      res.append(ent["text"])
  for _, ent in covid_raw_json["ref_entries"].items():
    res.append(ent["text"])
  assert len(res) > 0, "Failed to get expected texts"
  return res

def test_parse_success():
  "Ensures that parsing happens without failure"
  record = parse_covid_json.json_path_to_record(COVID_JSON)
  document_record.assert_valid_document_record(record)

def test_parse_title():
  record = parse_covid_json.json_path_to_record(COVID_JSON)
  record["title"] = covid_raw_json["metadata"]["title"]

def test_parse_text():
  record = parse_covid_json.json_path_to_record(COVID_JSON)
  actual_texts = [td["text"] for td in record["text_data"]]
  expected_texts = get_expected_texts()
  assert set(actual_texts) == set(expected_texts)

def test_parse_id():
  record = parse_covid_json.json_path_to_record(COVID_JSON)
  actual_id = record["pmid"]
  expected_id = covid_raw_json["paper_id"]
  assert actual_id == expected_id

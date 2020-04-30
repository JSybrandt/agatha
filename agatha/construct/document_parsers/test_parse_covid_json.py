from agatha.construct.document_parsers import parse_covid_json, document_record
import json

from pathlib import Path

json_data = []
json_paths = []
def load_json():
  if len(json_data) == 0:
    path_1 = Path("./test_data/example_covid_1.json")
    assert path_1.is_file(), f"Failed to find testing data: {path_1}"
    json_paths.append(path_1)
    with open(path_1, 'rb') as json_file:
      json_data.append(json.load(json_file))
    path_2 = Path("./test_data/example_covid_2.json")
    assert path_2.is_file(), f"Failed to find testing data: {path_2}"
    json_paths.append(path_2)
    with open(path_2, 'rb') as json_file:
      json_data.append(json.load(json_file))

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
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[0])
  document_record.assert_valid_document_record(record)

def test_parse_success_2():
  "Ensures that parsing happens without failure"
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[1])
  document_record.assert_valid_document_record(record)

def test_parse_title_1():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[0])
  record["title"] = json_data[0]["metadata"]["title"]

def test_parse_title_2():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[1])
  record["title"] = json_data[1]["metadata"]["title"]

def test_parse_text_1():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[0])
  actual_texts = [td["text"] for td in record["text_data"]]
  expected_texts = get_expected_texts(json_data[0])
  assert set(actual_texts) == set(expected_texts)

def test_parse_text_2():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[1])
  actual_texts = [td["text"] for td in record["text_data"]]
  expected_texts = get_expected_texts(json_data[1])
  assert set(actual_texts) == set(expected_texts)

def test_parse_id_1():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[0])
  actual_id = record["pmid"]
  expected_id = json_data[0]["paper_id"]
  assert actual_id == expected_id

def test_parse_id_2():
  load_json()
  record = parse_covid_json.json_path_to_record(json_paths[1])
  actual_id = record["pmid"]
  expected_id = json_data[1]["paper_id"]
  assert actual_id == expected_id

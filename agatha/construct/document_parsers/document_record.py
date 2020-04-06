from agatha.util.misc_util import Record
def new_document_record()->Record:
  return dict(
    pmid=None,
    version=None,
    date=None,
    language=None,
    medline_status=None,
    text_data=[],
    publication_types=[],
    mesh_headings=[],
    data_banks=[],
    authors=[],
  )


def assert_valid_document_record(record:Record)->None:
  def assert_field_name(name):
    assert name in record, f"Document {record} missing {name}"
  def assert_optional_string(name):
    assert_field_name(name)
    val = record[name]
    assert val is None or type(val) is str, \
        f"Document field {name} : {val} is not None or str"
  def assert_list(name):
    assert_field_name(name)
    assert type(record[name]) is list, \
        f"Document field {name}: {record[name]} is not List"

  new_rec = new_document_record()
  for name, blank_val in new_rec.items():
    if blank_val is None:
      assert_optional_string(name)
    elif blank_val == []:
      assert_list(name)
    else:
      raise Exception("Invalid new document record.")

  # do a special text_data test, already tested that it exists and is a list
  for text_elem in record["text_data"]:
    assert "text" in text_elem, f"Missing text field from text_data of {record}"
    assert "type" in text_elem, f"Missing type field from text_data of {record}"

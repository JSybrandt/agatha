from pathlib import Path
from agatha.util import umls_util

TEST_MRCONSO_PATH = Path("./test_data/tiny_MRCONSO.RRF")

"""
tiny_MRCONSO

Only contains the top-1000 lines from 2020AA
Only contains the following UMLS terms:


"""

EXPECTED_CUIS = {
    "C0000005", "C0000039", "C0000052", "C0000074", "C0000084", "C0000096",
    "C0000097", "C0000098", "C0000102", "C0000103", "C0000107", "C0000119",
    "C0000120", "C0000132", "C0000137", "C0000139", "C0000151", "C0000152",
    "C0000163", "C0000165", "C0000167", "C0000172", "C0000173", "C0000176",
    "C0000184", "C0000189", "C0000190", "C0000194", "C0000204", "C0000215",
}

def test_parse_mrconso():
  "Need to parse all 1000 lines of TEST_MRCONSO_PATH"
  lines = list(umls_util.parse_mrconso(TEST_MRCONSO_PATH))
  # Need 100 lines
  assert len(lines) == 1000
  # need all lines to have all fieldnames
  for line in lines:
    for fieldname in umls_util.MRCONSO_FIELDNAMES:
      assert fieldname in line
  # Need to have all UMLS terms
  actual_cuis = set(map(lambda x:x["cui"], lines))
  assert actual_cuis == EXPECTED_CUIS

def test_parse_first_line():
  actual = next(umls_util.parse_mrconso(TEST_MRCONSO_PATH))
  expected = {
    "cui": "C0000005",
    "lat": "ENG",
    "ts": "P",
    "lui": "L0000005",
    "stt": "PF",
    "sui": "S0007492",
    "ispref": "Y",
    "aui": "A26634265",
    "saui": "",
    "scui": "M0019694",
    "sdui": "D012711",
    "sab": "MSH",
    "tty": "PEP",
    "code": "D012711",
    "str": "(131)I-Macroaggregated Albumin",
    "srl": "0",
    "suppress": "N",
    "cvf": "256",
  }
  assert actual == expected

def test_filter_atoms_language_eng():
  atoms = list(umls_util.parse_mrconso(TEST_MRCONSO_PATH))
  # tiny MRCONSO has 337 English atoms
  filtered_atoms = umls_util.filter_atoms(
      mrconso_data=atoms,
      include_suppressed=True,
      filter_language="ENG"
  )
  num_items = len(list(filtered_atoms))
  assert num_items > 0
  assert num_items < len(atoms)
  for atom in filtered_atoms:
    assert atom["lat"] ==  "ENG"

def test_filter_atoms_suppress_content():
  atoms = list(umls_util.parse_mrconso(TEST_MRCONSO_PATH))
  # tiny MRCONSO has 858 unsurpassed atoms
  filtered_atoms = umls_util.filter_atoms(
      mrconso_data=atoms,
      include_suppressed=False,
      filter_language=None,
  )
  num_items = len(list(filtered_atoms))
  assert num_items > 0
  assert num_items < len(atoms)
  for atom in filtered_atoms:
    atom["supressed"] == "N"


def test_create_umls_index():
  umls_index = umls_util.UmlsIndex(TEST_MRCONSO_PATH)
  assert umls_index.num_codes() == len(EXPECTED_CUIS)
  assert umls_index.codes() == set(EXPECTED_CUIS)

def test_create_umls_index_filter():
  umls_index = umls_util.UmlsIndex(
      TEST_MRCONSO_PATH,
      include_suppressed=False,
      filter_language="ENG"
  )
  assert umls_index.num_codes() == len(EXPECTED_CUIS)
  assert umls_index.codes() == set(EXPECTED_CUIS)

def test_find_codes():
  umls_index = umls_util.UmlsIndex(
      TEST_MRCONSO_PATH,
      include_suppressed=False,
      filter_language="ENG"
  )
  umls_index.find_codes_with_pattern(
      r"^Dipalmitoylphosphatidylcholine*"
  ) == {"C0000039"}
  umls_index.find_codes_with_pattern(r".*Macro.*") == {"C0000005"}
  umls_index.find_codes_with_pattern(
      r".dip.*"
  ) == {"C0000005", "C0000039", "C0000194"}

def test_has_pref_text():
  umls_index = umls_util.UmlsIndex(
      TEST_MRCONSO_PATH,
      include_suppressed=False,
      filter_language="ENG"
  )
  for code in EXPECTED_CUIS:
    assert umls_index.contains_pref_text_for_code(code)

def test_has_code():
  umls_index = umls_util.UmlsIndex(
      TEST_MRCONSO_PATH,
      include_suppressed=False,
      filter_language="ENG"
  )
  for code in EXPECTED_CUIS:
    assert umls_index.contains_code(code)

def test_codes_with_minimum_edit_distance():
  umls_index = umls_util.UmlsIndex(
      TEST_MRCONSO_PATH,
      include_suppressed=False,
      filter_language="ENG"
  )
  text = "Dipalmitoylphosphatidylcholine"
  actual = umls_index.find_codes_with_close_text(text, ignore_case=True)
  assert actual == {"C0000039"}

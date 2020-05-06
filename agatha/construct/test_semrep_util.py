from agatha.construct import semrep_util
from pathlib import Path
import pytest
import lxml

# If SemRep isn't present, don't bother with these tests
SEMREP_INSTALL_DIR = Path("externals/semrep/2020/public_semrep")
METAMAP_INSTALL_DIR = Path("externals/semrep/2020/public_mm")
TEST_DATA_PATH = Path("test_data/semrep_input.txt")
TEST_COVID_DATA_PATH = Path("test_data/semrep_covid_input.txt")
TEST_COVID_XML_PATH = Path("test_data/semrep_covid.xml")

RUN_SEMREP_TESTS = SEMREP_INSTALL_DIR.is_dir() and METAMAP_INSTALL_DIR.is_dir()

def test_get_all_paths():
  "Tests that getting semrep paths gets all needed paths"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_install_dir=METAMAP_INSTALL_DIR
    )
    assert "metamap_install_dir" in paths
    assert "metamap_pos_server_path" in paths
    assert "metamap_wsd_server_path" in paths
    assert "semrep_install_dir" in paths
    assert "semrep_lib_dir" in paths
    assert "semrep_preamble_path" in paths
    assert "semrep_bin_path" in paths

def test_semrep_paths():
  "Tests that if we just need the semrep paths, we can get those"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        semrep_install_dir=SEMREP_INSTALL_DIR,
    )
    assert "semrep_install_dir" in paths
    assert "semrep_lib_dir" in paths
    assert "semrep_preamble_path" in paths
    assert "semrep_bin_path" in paths

def test_get_metamap_paths():
  "Tests that getting semrep paths gets all needed paths"
  if RUN_SEMREP_TESTS:
    paths = semrep_util.get_paths(
        metamap_install_dir=METAMAP_INSTALL_DIR
    )
    assert "metamap_install_dir" in paths
    assert "metamap_pos_server_path" in paths
    assert "metamap_wsd_server_path" in paths

def test_get_semrep_paths_fails():
  "Tests that if you give semrep paths bad install locations, it fails"
  if RUN_SEMREP_TESTS:
    with pytest.raises(AssertionError):
      semrep_util.get_paths(SEMREP_INSTALL_DIR, Path("."))
    with pytest.raises(AssertionError):
      semrep_util.get_paths(Path("."), METAMAP_INSTALL_DIR)
    with pytest.raises(AssertionError):
      semrep_util.get_paths(Path("."), Path("."))
    with pytest.raises(AssertionError):
      semrep_util.get_paths(semrep_install_dir=Path("."))
      semrep_util.get_paths(metamap_install_dir=Path("."))

def test_metamap_server():
  "Tests that we can actually run metamap"
  if RUN_SEMREP_TESTS:
    metamap_server = semrep_util.MetaMapServer(METAMAP_INSTALL_DIR)
    assert not metamap_server.running()
    metamap_server.start()
    assert metamap_server.running()
    metamap_server.stop()
    assert not metamap_server.running()

def test_another_metamap_server():
  """
  Tests that we can actually run a second metamap. The first might not have
  released ports.
  """
  if RUN_SEMREP_TESTS:
    metamap_server = semrep_util.MetaMapServer(METAMAP_INSTALL_DIR)
    assert not metamap_server.running()
    metamap_server.start()
    assert metamap_server.running()
    metamap_server.stop()
    assert not metamap_server.running()

def test_run_semrep():
  if RUN_SEMREP_TESTS:
    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=semrep_util.MetaMapServer(METAMAP_INSTALL_DIR),
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    output_file = Path("/tmp/test_run_semrep.xml")
    if output_file.is_file():
      output_file.unlink()
    assert not output_file.exists()
    runner.run(TEST_DATA_PATH, output_file)
    assert output_file.is_file()

def test_run_semrep_covid():
  if RUN_SEMREP_TESTS:
    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=semrep_util.MetaMapServer(METAMAP_INSTALL_DIR),
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    output_file = Path("/tmp/test_run_semrep_covid.xml")
    if output_file.is_file():
      output_file.unlink()
    assert not output_file.exists()
    runner.run(TEST_COVID_DATA_PATH, output_file)
    assert output_file.is_file()

def test_sentence_to_semrep_input():
  # We can run this test if SemRep is not installed
  sentences = [
      dict(id=1, sent_text="Sentence 1"),
      dict(id=2, sent_text="Sentence 2"),
  ]

  actual = semrep_util.sentence_to_semrep_input(sentences)
  expected = ["1|Sentence 1", "2|Sentence 2"]
  assert actual == expected

def test_sentence_to_semrep_input_filter_newline():
  # We can run this test if SemRep is not installed
  sentences = [
      dict(id=1, sent_text="Sentence\n1"),
      dict(id=2, sent_text="Sentence\n2"),
  ]

  actual = semrep_util.sentence_to_semrep_input(sentences)
  expected = ["1|Sentence 1", "2|Sentence 2"]
  assert actual == expected

def test_semrep_xml_to_records():
  "Ensures that parsing xml files happens without error"
  predicates = semrep_util.semrep_xml_to_records(TEST_COVID_XML_PATH)
  assert len(predicates) > 0

def test_semrep_id_to_agatha_sentence_id():
  expected = "s:12345:1:12"
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.E2"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.P11"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:12345:1:12.tx.2"
  ) == expected

def test_semrep_id_to_agatha_sentence_id_weird_id():
  expected = "s:abcd123efg567:1:12"
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.E2"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.P11"
  ) == expected
  assert semrep_util._semrep_id_to_agatha_sentence_id(
      "Ds:abcd123efg567:1:12.tx.2"
  ) == expected


def test_parse_semrep_xml_entity():
  raw_xml_entity = """
    <Entity
      id="Ds:32353859:1:6.E1"
      cui="C1517331"
      name="Further"
      semtypes="spco"
      text="Further"
      score="888"
      negated="false"
      begin="0"
      end="7"
    />
  """
  xml_entity = lxml.etree.fromstring(raw_xml_entity)
  expected = dict(
      id="s:32353859:1:6",
      cui="C1517331",
      name="Further",
      score=888,
      negated=False,
      begin=0,
      end=7,
  )
  assert semrep_util._parse_semrep_xml_entity(xml_entity) == expected

def test_parse_semrep_xml_predication():
  raw_xml_predication = """
  <Predication id="Ds:123:1:2.P1" negated="true" inferred="false">
   <Subject maxDist="5" dist="1" entityID="Ds:123:1:2.E1" relSemType="topp" />
   <Predicate type="USES" indicatorType="PREP" begin="1" end="2" />
   <Object maxDist="3" dist="1" entityID="Ds:123:1:2.E2" relSemType="aapp" />
  </Predication>
  """
  xml_predication= lxml.etree.fromstring(raw_xml_predication)
  semrepid2entity = {
      "Ds:123:1:2.E1": {
        "cui": "C1",
        "name": "First",
        "score": 888,
        "negated": False,
        "begin": 0,
        "end": 7,
      },
      "Ds:123:1:2.E2": {
        "cui": "C2",
        "name": "Second",
        "score": 888,
        "negated": False,
        "begin": 9,
        "end": 13,
      },
  }
  expected = {
      "negated": True,
      "inferred": False,
      "subject": {
        "cui": "C1",
        "name": "First",
        "score": 888,
        "negated": False,
        "begin": 0,
        "end": 7,
        "maxDist": 5,
        "dist": 1,
        "relSemType": "topp",
      },
      "predicate": {
        "type": "USES",
        "indicatorType": "PREP",
        "begin": 1,
        "end": 2,
      },
      "object": {
        "cui": "C2",
        "name": "Second",
        "score": 888,
        "negated": False,
        "begin": 9,
        "end": 13,
        "maxDist": 3,
        "dist": 1,
        "relSemType": "aapp",
      },
  }
  assert semrep_util._parse_semrep_xml_predication(
      xml_predication, semrepid2entity
  ) == expected


def test_parse_semrep_end_to_end():
  # Run SemRep
  if RUN_SEMREP_TESTS:
    records = [
        {
          "id": "s:1234:1:2",
          "sent_text": "Tobacco causes cancer in mice."
        },
        {
          "id": "s:2345:1:2",
          "sent_text": "Tobacco causes cancer in humans."
        },
    ]

    tmp_semrep_input = Path("/tmp/test_parse_semrep_end_to_end_input")
    tmp_semrep_output = Path("/tmp/test_parse_semrep_end_to_end_output")
    if tmp_semrep_input.is_file():
      tmp_semrep_input.unlink()
    if tmp_semrep_output.is_file():
      tmp_semrep_output.unlink()

    with open(tmp_semrep_input, 'w') as semrep_input_file:
      for line in semrep_util.sentence_to_semrep_input(records):
        semrep_input_file.write(f"{line}\n")

    runner = semrep_util.SemRepRunner(
        semrep_install_dir=SEMREP_INSTALL_DIR,
        metamap_server=semrep_util.MetaMapServer(METAMAP_INSTALL_DIR),
        lexicon_year=2020,
        mm_data_year="2020AA",
    )
    runner.run(tmp_semrep_input, tmp_semrep_output)
    assert tmp_semrep_output.is_file()

    # should return one per document
    records = semrep_util.semrep_xml_to_records(tmp_semrep_output)
    assert len(records) == 2

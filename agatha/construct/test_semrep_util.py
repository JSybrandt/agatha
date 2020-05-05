from agatha.construct import semrep_util
from pathlib import Path
import pytest

# If SemRep isn't present, don't bother with these tests
SEMREP_INSTALL_DIR = Path("externals/semrep/2020/public_semrep")
METAMAP_INSTALL_DIR = Path("externals/semrep/2020/public_mm")
TEST_DATA_PATH = Path("test_data/semrep_input.txt")
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

"""SemRep Dask Utilities

This module helps run SemRep within the Agatha graph construction pipeline.
For this to work, we need to run SemRep on each machine in our cluster, and
extract all necessary information as edges.

To run SemRep, you must first start the MetaMap servers for part-of-speech
tagging and word-sense disambiguation. These are supplied through MetaMap.
Specifically, we are expecting to find `skrmedpostctl` and `wsdserverctl` in
the directory specified through `config.semrep.metamap_bin_dir`. Once these
servers are started we are free to run `semrep`.

"""

from pathlib import Path
from typing import Dict, Tuple, List
import agatha.construct.dask_process_global as dpg
from multiprocessing import Process
import subprocess
import os

def get_paths(
    semrep_install_dir:Path=None,
    metamap_install_dir:Path=None,
)->Dict[str, Path]:
  """Looks up all of the necessary files needed to run SemRep.

  This function identifies the binaries and libraries needed to run SemRep.
  Additionally, this function asserts that all the needed files are actually
  present.

  This function will find:
  `skrmedpostctl`: Metamap's SKR/Medpost Part-of-Speech Tagger Server
  `wsdserverctl`: Metamap's Word Sense Disambiguation (WSD) Server
  `SEMREPrun.v*`: The preamble needed to run SemRep
  `semrep.v*.BINARY.Linux`: The binary used to run SemRep
  `lib`: The Java libraries in SemRep

  If only one or the other semrep_install_dir or metamap_install_dir is
  specified, then only that components paths will be returned.

  Args:
    semrep_install_dir: The install location of SemRep. Named `public_semrep` by
      default.
    metamap_install_dir: The install location of MetaMap. Named `public_mm` my
      default.

  Returns:
    A dictionary of names and associated paths. If a name ends in `_path` then
    it has been asserted `is_file()`. If name ends in `_dir` it has been
    asserted `is_dir()`.

  """

  def is_dir(d):
    assert d.is_dir(), f"Failed to find directory: {d.absolute()}"
  def is_file(f):
    assert f.is_file(), f"Failed to find file: {f.absolute()}"
  def match(d, pattern):
    is_dir(d)
    # drop "*.in" files
    files = [f for f in d.glob(pattern) if not f.suffix == ".in"]
    assert len(files) != 0, f"No file matching {pattern} in {d.absolute()}"
    assert len(files) == 1, f"Ambiguous match of {pattern} in {d.absolute()}"
    return files[0]

  res = {}
  if metamap_install_dir is not None:
    metamap_install_dir = Path(metamap_install_dir)
    is_dir(metamap_install_dir)
    res["metamap_install_dir"] = metamap_install_dir
    metamap_pos_server_path = (
        metamap_install_dir
        .joinpath("bin")
        .joinpath("skrmedpostctl")
    )
    is_file(metamap_pos_server_path)
    res["metamap_pos_server_path"] = metamap_pos_server_path
    metamap_wsd_server_path = (
        metamap_install_dir
        .joinpath("bin")
        .joinpath("wsdserverctl")
    )
    is_file(metamap_wsd_server_path)
    res["metamap_wsd_server_path"] = metamap_wsd_server_path
  if semrep_install_dir is not None:
    semrep_install_dir = Path(semrep_install_dir)
    is_dir(semrep_install_dir)
    res["semrep_install_dir"] = semrep_install_dir
    semrep_lib_dir = semrep_install_dir.joinpath("lib")
    is_dir(semrep_lib_dir)
    res["semrep_lib_dir"] = semrep_lib_dir
    semrep_preamble_path = match(
        semrep_install_dir.joinpath("bin"),
        "SEMREPrun.v*"
    )
    res["semrep_preamble_path"] = semrep_preamble_path
    semrep_bin_path = match(
        semrep_install_dir.joinpath("bin"),
        "semrep.v*.BINARY.Linux"
    )
    res["semrep_bin_path"] = semrep_bin_path
  return res

################################################################################
# Per-Node Server ##############################################################
################################################################################

class MetaMapServer():
  """Manages connection to MetaMap

  SemRep requires a connection to MetaMap. This means we need to launch the
  pos_server and wsd_server. This class is responsible for managing that server
  connection. We anticipate using one server per-worker, meaning this class
  will be initialized using  dask_process_global initializer.

  Args:
    metamap_install_dir: The install location of MetaMap

  """
  def __init__(self, metamap_install_dir:Path):
    # Get paths to pos and wsd servers
    self.pos_server_proc = None
    self.wsd_server_proc = None
    paths = get_paths(metamap_install_dir=metamap_install_dir)
    self.metamap_pos_server_path = paths["metamap_pos_server_path"]
    self.metamap_wsd_server_path = paths["metamap_wsd_server_path"]

  def __del__(self):
    self.stop()

  def start(self):
    "Call to start the MetaMap servers, if not already running."
    if not self.running():
      self.pos_server_proc = Process(
          target=self.metamap_pos_server_path,
          args=["start"]
      )
      self.wsd_server_proc = Process(
          target=self.metamap_wsd_server_path,
          args=["start"]
      )
      self.pos_server_proc.start()
      self.wsd_server_proc.start()

  def stop(self):
    "Stops the MetaMap servers, if running"
    if self.pos_server_proc is not None:
      self.pos_server_proc.kill()
      self.pos_server_proc = None
    if self.wsd_server_proc is not None:
      self.wsd_server_proc.kill()
      self.wsd_server_proc = None

  def running(self):
    "Returns True if `start` was called"
    return self.pos_server_proc is not None and self.wsd_server_proc is not None

def get_metamap_server_initializer(
    metamap_install_dir:Path,
)->Tuple[str, dpg.Initializer]:
  def _init():
    return MetaMapServer(metamap_install_dir)
  return f"semrep:metamap_server", _init

class SemRepRunner():
  """Responsible for running SemRep.

  Given a metamap server and additional SemRep Configs, this class actually
  processes text and generates predicates. All SemRep predicates are copied
  here and provided through the constructor. All defaults are preserved.

  Args:
    semrep_install_dir: Location where semrep is installed.
    metamap_server: A connection to the MetaMapServer that enables us to
      actually run SemRep. We use this to ensure server is running.
    work_dir: Location to store intermediate files used to communicate with
      SemRep.
    anaphora_resolution: SemRep Flag
    dysonym_processing: SemRep Flag
    lexicon_year: The year as an int which we use with MetaMap. Ex: 2020
    mm_data_version: Specify which UMLS data version. Ex: USAbase
    mm_data_year: Specify UMLS release year. Ex: 2020AA
    relaxed_model: SemRep Flag
    use_generic_domain_extensions: SemRep Flag
    use_generic_domain_modification: SemRep Flag
    word_sense_disambiguation: SemRep Flag

  """
  def __init__(
      self,
      semrep_install_dir:Path,
      metamap_server:MetaMapServer,
      # SemRep Flags
      anaphora_resolution=True,
      dysonym_processing=True,
      lexicon_year:int=2006,
      mm_data_version:str="USAbase",
      mm_data_year:str="2006AA",
      relaxed_model:bool=True,
      single_line_delim_input_w_id=True,
      use_generic_domain_extensions=False,
      use_generic_domain_modification=False,
      word_sense_disambiguation=True,
  ):
    # set paths
    paths = get_paths(semrep_install_dir=semrep_install_dir)
    self.semrep_install_dir = paths["semrep_install_dir"]
    self.semrep_lib_dir = paths["semrep_lib_dir"]
    self.semrep_preamble_path = paths["semrep_preamble_path"]
    self.semrep_bin_path = paths["semrep_bin_path"]
    # Set serer
    self.metamap_server=metamap_server

    self.anaphora_resolution = anaphora_resolution
    self.dysonym_processing = dysonym_processing
    self.lexicon_year = lexicon_year
    self.mm_data_version= mm_data_version
    self.mm_data_year = mm_data_year
    self.relaxed_model = relaxed_model
    self.single_line_delim_input_w_id = single_line_delim_input_w_id
    self.use_generic_domain_extensions = use_generic_domain_extensions
    self.use_generic_domain_modification = use_generic_domain_modification
    self.word_sense_disambiguation = word_sense_disambiguation

  def _get_env(self)->Dict[str,str]:
    "Adds the necessary semrep_lib_dir to LD_LIBRARY_PATH"
    env = os.environ.copy()
    if "LD_LIBRARY_PATH" in env:
      env["LD_LIBRARY_PATH"] += f":{self.semrep_lib_dir}"
    else:
      env["LD_LIBRARY_PATH"] = str(self.semrep_lib_dir)
    return env

  def _get_flags(self, input_path:Path, output_path:Path)->List[str]:
    "Gets flags for running semrep"
    res = []
    res.append(str(self.semrep_preamble_path))
    res.append(str(self.semrep_bin_path))
    if self.anaphora_resolution:
      res.append("--anaphora_resolution")
    if self.dysonym_processing:
      res.append("--dysonym_processing")
    res.append("--lexicon_year")
    res.append(str(self.lexicon_year))
    res.append("--mm_data_version")
    res.append(self.mm_data_version)
    res.append("--mm_data_year")
    res.append(self.mm_data_year)
    if self.relaxed_model:
      res.append("--relaxed_model")
    if self.single_line_delim_input_w_id:
      res.append("--sldiID")
    if self.use_generic_domain_extensions:
      res.append("--use_generic_domain_extensions")
    if self.use_generic_domain_modification:
      res.append("--use_generic_domain_modification")
    if self.word_sense_disambiguation:
      res.append("--word_sense_disambiguation")
    res.append("--xml_output_format")
    res.append(str(input_path))
    res.append(str(output_path))
    return res

  def run(self, input_path:Path, output_path:Path)->None:
    """Actually calls SemRep with an input file.

    Args:
      input_path: The location of the SemRep Input file

    Returns:
      The path produced by SemRep representing XML output.

    """
    if not self.metamap_server.running():
      self.metamap_server.start()
    input_path = Path(input_path)
    assert input_path.is_file(), f"Failed to find {input_path}"
    assert not output_path.exists(), f"Refusing to overwrite {output_path}"
    subprocess.run(
        self._get_flags(input_path, output_path),
        env=self._get_env()
    )
    assert output_path.is_file(), f"SemRep Failed to produce {output_path}"

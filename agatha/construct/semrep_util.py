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

import agatha.construct.dask_process_global as dpg
from agatha.construct import text_util
from agatha.util.misc_util import Record
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
from typing import Dict, Tuple, List, Iterable, Callable, Any
import lxml
import lxml.etree
from agatha.util.entity_types import SENTENCE_TYPE
import re
from copy import deepcopy

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


################################################################################
# Dask Utility Functions #######################################################
################################################################################

def sentence_to_semrep_input(records:Iterable[Record])->List[str]:
  """Processes Sentence Records for SemRep Input

  The SemRepRunner, with the default single_line_delim_input_w_id flag set,
  expects input in the form:
  ```
  id1|Sentence 1
  id2|Sentence 2
    ...
  ```

  This function converts Agatha sentence records, containing the `sent_text` and
  `id` fields into the single_line_delim_input_w_id format. Because each
  sentence must occur on its own line, this function will replace newline
  characters with spaces in output.

  Recommend Usage:

  ```python3
  sentences.map_partitions(sentence_to_semrep_input).to_textfiles(...)
  ```

  Args:
    records: Sentence records, each containing `sent_text` and `id`

  """
  res = []
  for record in records:
    assert "sent_text" in record, "Record missing sent_text field"
    assert "id" in record, "Record missing id field"
    text = str(record["sent_text"])
    id_ = str(record["id"])
    # Don't want newlines in text
    text = text.replace("\n", " ")
    # Don't want pipe in id
    assert "|" not in id_, "SemRep IDs cannot contain pipe character."
    res.append(f"{id_}|{text}")
  return res


def _semrep_id_to_agatha_sentence_id(semrep_id:str)->str:
  """Cleans the SemRep ID for Agatha

  When running SemRep following `sentence_to_semrep_input`, the `id` attribute
  of each xml element will help cross-reference the XML results to the rest of
  the data for each sentence. However, SemRep will pre-pend a 'D'. For entities
  and predicates, a suffix of `.E#` or `.P#` will also be added. Utterances
  receive suffixes like `.tx.1` based on the count and type.

  Args:
    semrep_id: An id like: "Ds:32353859:1:6", "Ds:32353859:1:6.E11",
      or "Ds:32353859:1:6.tx.1".

  Returns:
    A cleaned id, like "s:32353859:1:6"
  """
  assert semrep_id[0] == "D", "Invalid semrep id. Expected to start with D"
  # This will match a string that starts with a properly formatted agatha
  # sentence ID. Note, CORD-19 ids, such as those for PMC articles, may have
  # any alphanumeric id instead of a numeric pmid
  regex_starts_with_sentence_id = "^s:[a-zA-Z0-9]+:[0-9]+:[0-9]+"
  # We already established that the first character is "D". Now the rest
  # of the string needs to start with the sentence_id
  found_pattern = re.search(regex_starts_with_sentence_id, semrep_id[1:])
  assert found_pattern is not None, f"Invalid sentence id: {semrep_id}"
  return found_pattern.group(0)

def _str_to_bool(s:str)->bool:
  "Converts 'true' and 'false' to True and False"
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Invalid bool string: {s}")

def _set_or_none(
    rec:Record,
    xml:lxml.etree._Element,
    attr:str,
    fn:Callable[[str], Any]=str,
)->None:
  """Sets rec[attr] = fn(xml.attrib[attr]) safely.

  If xml does not have attr, rec[attr] = None

  Args:
    rec: Record to set
    xml: Xml element with attributes we want to select
    attr: Name of xml attribute
    fn: Conversion from string. Defaults to str.
  """
  if attr in xml.attrib:
    rec[attr] = fn(xml.attrib[attr])
  else:
    rec[attr] = None

def _parse_semrep_xml_entity(xml_entity:lxml.etree._Element)->Record:
  "Collects attributes from SemRep Entities"
  res = {}
  _set_or_none(res, xml_entity, "id", _semrep_id_to_agatha_sentence_id)
  _set_or_none(res, xml_entity, "cui", str)
  _set_or_none(res, xml_entity, "name", str)
  _set_or_none(res, xml_entity, "score", int)
  _set_or_none(res, xml_entity, "negated", _str_to_bool)
  _set_or_none(res, xml_entity, "begin", int)
  _set_or_none(res, xml_entity, "end", int)
  return res

def _parse_semrep_xml_predication(
    xml_predication:lxml.etree._Element,
    semrepid2entity:Dict[str, Record],
)->Record:
  """Parses a predication object, and dereferences entity ids.

  Example:
  ```
  <Predication id="Ds:123:1:2.P1" negated="true" inferred="false">
   <Subject maxDist="5" dist="1" entityID="Ds:123:1:2.E5" relSemType="topp" />
   <Predicate type="USES" indicatorType="PREP" begin="93" end="96" />
   <Object maxDist="3" dist="1" entityID="Ds:123:1:2.E4" relSemType="aapp" />
  </Predication>
  ```

  Becomes:
  ```
  {
    "id": s:123:1:2,
    "negated"=True,
    "inferred"=False,
    "subject": {
      "cui": "C0199176",
      "name": "Prophylactic treatment",
      "semtypes": "topp",
      "text": "prevention",
      "score": 1000,
      "negated": False,
      "begin": 101,
      "end": 111,
    }
    "predicate": {
      "type": "USES",
      "indicatorType": "PREP",
      "begin": 93,
      "end": 96,
    }
    "object": {
      ...
    }
  }
  ```
  """
  xml_subject = xml_predication.find("Subject")
  xml_pred = xml_predication.find("Predicate")
  xml_object = xml_predication.find("Object")
  assert xml_subject is not None, "Predication missing Subject"
  assert xml_pred is not None, "Predication missing Predicate"
  assert xml_object is not None, "Predication missing Object"

  def prep_entity(xml_entity):
    assert xml_entity.attrib["entityID"] in semrepid2entity, \
      f"Predicate references unknown entity: {xml_entity.attrib['entityID']}"
    ent =  deepcopy(semrepid2entity[xml_entity.attrib["entityID"]])
    _set_or_none(ent, xml_entity, "maxDist", int)
    _set_or_none(ent, xml_entity, "dist", int)
    _set_or_none(ent, xml_entity, "relSemType")
    return ent

  pred = {}
  _set_or_none(pred, xml_pred, "type")
  _set_or_none(pred, xml_pred, "indicatorType")
  _set_or_none(pred, xml_pred, "begin", int)
  _set_or_none(pred, xml_pred, "end", int)

  res = {}
  _set_or_none(res, xml_predication, "negated", _str_to_bool)
  _set_or_none(res, xml_predication, "inferred", _str_to_bool)
  res["subject"] = prep_entity(xml_subject)
  res["predicate"] = pred
  res["object"] = prep_entity(xml_object)
  return res


def semrep_xml_to_records(xml_path:Path)->List[Record]:
  """Parses SemRep XML records to produce Predicate Records

  This parses SemRep XML output, generated by SemRep v1.8 via the
  `--xml_output_format` flag. Take a look [here][1] to get more details on the
  XML spec. Additional details below. We specifically focus on parsing XML
  records produced by the SemRepRunner.

  XML Format Summary: The XML file starts with an overarching SemRepAnnotation
  object, containing multiple `Document` records, one per input text. These
  documents contain identified UMLS terms (`Document > Utterance > Entity`) and
  predicates (`Document > Utterance > Predication`). One document may have
  multiple utterances.

  Args:
    xml_path: Location of XML file to parse.

  Returns:
    A list of python dicts wherein each corresponds to a detected predicate.

  [1]:https://semrep.nlm.nih.gov/SemRep.v1.8_XML_output_desc.html

  """
  res = []
  xml_path = Path(xml_path)
  assert xml_path.is_file(), f"Failed to find semrep_xml file: {xml_path}"
  with open(xml_path, 'rb') as xml_file:
    # For each document. One document corresponds to one sentence
    for _, xml_doc in lxml.etree.iterparse(xml_file, tag="Document"):

      # document data
      semrepid2entity = {}
      predicates = []

      # For each sentence, typically there will only be one, unless the SemRep
      # sentence splitter makes a different decision than us
      xml_uttrs = xml_doc.findall("Utterance")
      if xml_uttrs is not None:
        for xml_uttr in xml_uttrs:

          # Collect the mentioned UMLS terms
          xml_ents = xml_uttr.findall("Entity")
          if xml_ents is not None:
            for xml_ent in xml_ents:
              semrepid2entity[xml_ent.attrib["id"]] = \
                  _parse_semrep_xml_entity(xml_ent)

          # Collect the identified predicates
          xml_preds = xml_uttr.findall("Predication")
          if xml_preds is not None:
            for xml_predication in xml_preds:
              predicates.append(
                  _parse_semrep_xml_predication(
                    xml_predication,
                    semrepid2entity
              ))
      res.append({
        "id": _semrep_id_to_agatha_sentence_id(xml_doc.attrib["id"]),
        "entities": list(semrepid2entity.keys()),
        "predicates": predicates
      })
  return res

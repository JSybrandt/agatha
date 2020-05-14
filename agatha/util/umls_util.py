"""umls_util.py

This module is responsible for cross referencing UMLS MRCONSO. This means that
we will be able to both lookup UMLS terms from plaintext descriptions, and
vice-versa.

"""

from collections import defaultdict
from pathlib import Path
from csv import DictReader
from typing import Dict, List, Optional, Iterable, Set
import re
import Levenshtein

# Columns of MRCONSO file in order
MRCONSO_FIELDNAMES = [
  "cui",       # Unique identifier for concept
  "lat",       # Language of term
  "ts",        # Term status
  "lui",       # Unique identifier for term
  "stt",       # String type
  "sui",       # Unique identifier for string
  "ispref",    # Atom status - preferred (Y) or not (N)
  "aui",       # Unique identifier for atom
  "saui",      # Source asserted atom identifier [optional]
  "scui",      # Source asserted concept identifier [optional]
  "sdui",      # Source asserted descriptor identifier [optional]
  "sab",       # Abbreviated source name (SAB)
  "tty",       # Abbreviation for term type in source vocabulary.
  "code",      # Most useful source asserted identifier
  "str",       # String
  "srl",       # Source restriction level
  "suppress",  # Suppressible flag. Values = O, E, Y, or N
  "cvf",       # Content View Flag
]

# https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#TS

# This url has some content on abbreviations

def atom_contains_all_fields(atom:Dict[str, str])->bool:
  for fieldname in MRCONSO_FIELDNAMES:
    if fieldname not in atom:
      return False
  return True

def parse_mrconso(mrconso_path:Path)->Iterable[Dict[str, str]]:
  """Parses MRCONSO file

  The MRCONSO file, as described in:

  https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/

  Has columns described in `umls_util.MRCONSO_FIELDNAMES`.

  This function takes each line of the MRCONSO.RRF file name parses out each
  field.  The result is a list of dictionaries, where `parse_mrconso(...)[i]`
  contains all of the fields of line `i`. For  instance, you can get the CUID
  of line `i` by calling `parse_mrconso(...)[i]['cui']`

  Args:
    mrconso_path: The filepath to MRCONSO.RRF. Must end in `.RRF`.

  Returns:
    List of parsed MRCONSO data. Each line contains the fields defined in
    MRCONSO_FIELDNAMES.

  """
  mrconso_path = Path(mrconso_path)
  assert mrconso_path.is_file(), f"Failed to find {mrconso_path}"
  assert mrconso_path.suffix.lower() == ".rrf", \
    f"File {mrconso_path} does not have `.RRF` extension."
  with open(mrconso_path, 'r', newline='') as mrconso_file:
    reader = DictReader(
        mrconso_file,
        fieldnames=MRCONSO_FIELDNAMES,
        delimiter="|"
    )
    for row in reader:
      # If there's extra data, I don't want to see it
      if None in row:
        del row[None]
      yield row

def filter_atoms(
    mrconso_data:List[Dict[str,str]],
    include_suppressed:bool=False,
    filter_language:Optional[str]="ENG",
)->Iterable[Dict[str,str]]:
  """Filters the lines of MRCONSO

  If `include_suppressed` is set, then atoms with
  `SUPPRESS` set will be included in the result.

  If `filter_language` is not `None`, then only atoms with `LAT` set to the
  filter language will be included.

  """
  if filter_language is not None:
    filter_language = filter_language.lower()

  for atom in mrconso_data:
    assert atom_contains_all_fields(atom), f"Filter passed invalid atom: {atom}"
    if (
        ( # unsupressed, or allowing supressed
          atom["suppress"].lower() == "n"
          or include_suppressed
        ) and ( # no language set, or the correct language
          filter_language is None
          or atom["lat"].lower() == filter_language
        )
    ):
      yield atom


class UmlsIndex():
  """
  The UmlsIndex is responsible for managing the MRCONSO file.

  When we create the UmlsIndex we create the intermediate data structures
  required to index all UMLS keywords, and all plaintext atoms. You can
  download a MRCONSO file associated with a UMLS release here:

  www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

  Take a look to see what the MRCONSO file format is supposed to look like:

  https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/

  Args:
    mrconso_path: The path to a MRCONSO RRF file.
    include_supressed_content: By default, this index will only consider terms
      that have not been marked as `SUPPRESS`. If this flag is set, we will
      include all terms.
    filter_language: If set, this index will only consider names appearing
      in the selected langauge (default = ENG). If set to `None`, all terms
      will be considered.

  """
  def __init__(
      self,
      mrconso_path:Path,
      **filter_kwargs
  ):
    self._code2preferred_text = {}
    self._code2texts = defaultdict(set)
    self._text2codes = defaultdict(set)
    for atom in filter_atoms(parse_mrconso(mrconso_path), **filter_kwargs):
      code = atom["cui"].upper()
      text = atom["str"]
      if atom["ispref"].lower() == "y":
        self._code2preferred_text[code] = text
      self._code2texts[code].add(text)
      self._text2codes[text].add(code)

  def codes(self)->Set[str]:
    return set(self._code2texts.keys())

  def num_codes(self)->int:
    return len(self._code2texts)

  def get_pref_text(self, code:str)->str:
    code = code.upper()
    assert code in self._code2preferred_text, \
        f"Failed to find pref text for {code}"
    return self._code2preferred_text[code]

  def get_texts(self, code:str)->Set[str]:
    code = code.upper()
    assert code in self._code2texts, \
        f"Failed to find text for {code}"
    return self._code2texts[code]

  def find_codes_with_pattern(self, pattern:str)->Set[str]:
    "Returns the set of codes with text that matches the regex pattern"
    res = set()
    for text, codes in self._text2codes.items():
      if re.match(pattern, text) is not None:
        res |= codes
    return res

  def contains_code(self, code:str)->bool:
    return code.upper() in self._code2texts

  def contains_pref_text_for_code(self, code:str)->bool:
    return code.upper() in self._code2preferred_text

  def find_codes_with_close_text(
      self,
      text:str,
      ignore_case:bool=False,
  )->Set[str]:
    """Returns the set of codes with text most similar to that provided.

    Each text field of all managed atoms is compared to the given text.
    The set of codes with text that minimize edit distance with the
    given text are returned.

    For example, if codes C1 and C2 are both equally distant to text, then
    both will be returned.
    """

    if ignore_case:
      text = text.lower()

    min_distance = None
    codes_at_min_distance = set()
    for code_text, codes in self._text2codes.items():
      if ignore_case:
        code_text = code_text.lower()
      dist = Levenshtein.distance(text, code_text)
      if min_distance is None or dist < min_distance:
        min_distance = dist
        codes_at_min_distance.clear()
      if min_distance == dist:
        codes_at_min_distance |= codes
    return codes_at_min_distance

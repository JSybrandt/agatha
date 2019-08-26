from lxml import etree
import gzip
from typing import Iterator
from pymoliere.datatypes.datatypes_pb2 import PubMedRecord
from pathlib import Path

def parse_pubmed_xml(xml_gz_path:Path)->Iterator[PubMedRecord]:
  assert xml_gz_path.is_file()
  assert str(xml_gz_path).endswith(".xml.gz")
  with gzip.open(str(xml_gz_path), "rb") as xml_file:
    for _, element in etree.iterparse(xml_file, tag="MedlineCitation"):
      record = PubMedRecord()

      pmid_elem = element.find("PMID")
      if pmid_elem is not None:
        record.id = int(pmid_elem.text)
        record.version = int(pmid_elem.attrib["Version"])

      try:
        article_elem = element.find("Article")
        if article_elem is not None:
          language_elem = article_elem.find("Language")
          if language_elem is not None:
            record.language = language_elem.text
          title_elem = article_elem.find("ArticleTitle")
          if title_elem is not None:
            record.raw_title = title_elem.text
          abstract_elem = article_elem.find("Abstract")
          if abstract_elem is not None:
            record.raw_abstract = "".join([
              x.text for x in abstract_elem.findall("AbstractText")
            ])
          author_list_elem = article_elem.find("AuthorList")
          if author_list_elem is not None:
            for author_elem in author_list_elem.findall("Author"):
              auth = record.authors.add()
              first_name_elem = author_elem.find("ForeName")
              if first_name_elem is not None:
                auth.first_name = first_name_elem.text
              last_name_elem = author_elem.find("LastName")
              if last_name_elem is not None:
                auth.last_name = last_name_elem.text
        yield record
      except:
        print("ERR:", record.id)


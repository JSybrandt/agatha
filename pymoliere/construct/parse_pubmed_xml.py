from lxml import etree
import gzip
from typing import Iterator
from pymoliere.datatypes.datatypes_pb2 import PubMedRecord
from pathlib import Path
import dask.dataframe as dask_df
import pandas as pd
from pymoliere.construct.file_util import (
    copy_to_local_scratch,
)
import pyarrow as pa

def xml_obj_to_date(elem)->str:
  year_elem = elem.find("Year")
  month_elem = elem.find("Month")
  day_elem = elem.find("Day")
  year = int(year_elem.text) if year_elem else 0
  month = int(month_elem.text) if month_elem else 0
  day = int(day_elem.text) if day_elem else 0
  return f"{year:04}-{month:02}-{day:02}"

def parse_pubmed_xml(
    shared_xml_gz_path:Path,
    local_scratch_dir:Path,
)->pd.DataFrame:
  "First, copies a given xml.gz file to local. Then processes."
  assert shared_xml_gz_path.is_file()
  assert str(shared_xml_gz_path).endswith(".xml.gz")
  local_xml_gz_path = copy_to_local_scratch(
      src=shared_xml_gz_path,
      local_scratch_dir=local_scratch_dir,
  )

  records = []
  with gzip.open(str(local_xml_gz_path), "rb") as xml_file:
    for _, pubmed_elem in etree.iterparse(xml_file, tag="PubmedArticle"):
      record = {
          "id": None,
          "version": None,
          "date": None,
          "language": None,
          "raw_title": None,
          "raw_abstract": None,
          "publication_types": [],
          "mesh_headings": [],
          "medline_status": None,
      }

      medline_cite_elem = pubmed_elem.find("MedlineCitation")
      if medline_cite_elem is not None:
        record["medline_status"] = medline_cite_elem.attrib["Status"]

        pmid_elem = medline_cite_elem.find("PMID")
        if pmid_elem is not None:
          record["id"] = int(pmid_elem.text)
          record["version"] = int(pmid_elem.attrib["Version"])

        article_elem = medline_cite_elem.find("Article")
        if article_elem is not None:
            language_elem = article_elem.find("Language")
            if language_elem is not None:
              record["language"] = language_elem.text

            title_elem = article_elem.find("ArticleTitle")
            if title_elem is not None:
              record["raw_title"] = title_elem.text

            abstract_elem = article_elem.find("Abstract")
            if abstract_elem is not None:
              text_elems = abstract_elem.findall("AbstractText")
              if text_elems is not None:
                record["raw_abstract"] = "".join([
                  x.text for x in text_elems if x.text is not None
                ])

            pub_type_list_elem = article_elem.find("PublicationTypeList")
            if pub_type_list_elem is not None:
              pub_type_elems = pub_type_list_elem.findall("PublicationType")
              if pub_type_elems is not None:
                for x in pub_type_elems:
                  if x.text is not None:
                    record["publication_types"].append(x.text)

        chemical_list_elem = medline_cite_elem.find("ChemicalList")
        if chemical_list_elem is not None:
          chemical_elems = chemical_list_elem.findall("Chemical")
          if chemical_elems is not None:
            for chem_elem in chemical_elems:
              name_elem = chem_elem.find("NameOfSubstance")
              if name_elem is not None:
                record["mesh_headings"].append(name_elem.attrib["UI"])

        mesh_heading_list_elem = medline_cite_elem.find("MeshHeadingList")
        if mesh_heading_list_elem is not None:
          mesh_heading_elems = mesh_heading_list_elem.find("MeshHeading")
          if mesh_heading_elems is not None:
            for mesh_elem in mesh_heading_elems:
              desc_elem = mesh_elem.find("DescriptorName")
              if desc_elem is not None:
                record["mesh_headings"].append(desc_elem.attrib["UI"])

      pubmed_data_elem = pubmed_elem.find("PubmedData")
      if pubmed_data_elem is not None:
        history_elem = pubmed_data_elem.find("History")
        if history_elem is not None:
          pm_date_elems = history_elem.findall("PubMedPubDate")
          if pm_date_elems is not None:
            for date_elem in pm_date_elems:
              date = xml_obj_to_date(date_elem)
              if date != "0000-00-00" and (
                  (record["date"] is None) or (record["date"] > date)
              ):
                record["date"] = date

      # Set defaults, or skip if required field not present
      if record["id"] is None:
        continue
      if record["version"] is None:
        record["version"] = -1
      if record["date"] is None:
        date = "0000-00-00"
      if record["language"] is None:
        record["language"] = "unknown"
      if record["raw_title"] is None:
        record["raw_title"] = ""
      if record["raw_abstract"] is None:
        record["raw_abstract"] = ""
      record["publication_types"] = ",".join([
        f'"{x}"' for x in record["publication_types"]
      ])
      record["mesh_headings"] = ",".join([
        f'"{x}"' for x in record["mesh_headings"]
      ])
      records.append(record)

  return pd.DataFrame(records)

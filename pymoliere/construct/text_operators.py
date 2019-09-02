from dask.distributed import Client
from pathlib import Path
from pymoliere.util.pipeline_operator import PipelineOperator
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker
from typing import List, Tuple, Any, Optional
import dask.dataframe as ddf
import pandas as pd
import spacy

def setup_scispacy(
    scispacy_version:str,
    extra_parts:bool=False,
)->Tuple[Any, UmlsEntityLinker]:
  print("Loading scispacy... Might take a bit.")
  nlp = spacy.load(scispacy_version)

  if extra_parts:
    # Add the abbreviation pipe to the spacy pipeline.
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
    # Add UMLS linker to pipeline
    umls_linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(umls_linker)
    return nlp, umls_linker
  else:
    return nlp, None

def text_fields_to_sentences(
    dataframe:pd.DataFrame,
    text_fields:List[str],
    sentence_text_field:str,
    sentence_idx_field:str,
    scispacy_version:Optional[str]=None,
    nlp:Optional[Any]=None,
)->pd.DataFrame:

  # Need to load cached value.
  if nlp is None:
    nlp, _ = setup_scispacy(scispacy_version)

  new_col_order = list(dataframe.columns.drop(text_fields))
  new_col_order += [sentence_idx_field, sentence_text_field]

  def fields_to_sentences(row):
    result = []
    non_text_data = row.drop(text_fields)
    for text_field in row[text_fields]:
      doc = nlp(text_field)
      for sentence_tokens in doc.sents:
        sentence_text = "".join([t.string for t in sentence_tokens])
        r = non_text_data.copy()
        r[sentence_text_field] = sentence_text
        r[sentence_idx_field] = len(result)
        result.append(r)
    return result
  series = dataframe.apply(
    fields_to_sentences,
    axis=1
  ).explode()
  return pd.DataFrame(
      [s[new_col_order].values for s in series],
      columns=new_col_order
  )


class SplitSentencesOperator(PipelineOperator):
  def __init__(
      self,
      input_dataset:str,
      scispacy_version:str,
      text_fields:List[str],
      sentence_text_field:str="sentence",
      sentence_idx_field:str="sentence_idx",
      **kwargs
  ):
    """
    For each row in the input dataframe, consider all provided text fields.
    Then produce a new dataframe with one row per sentence.  All fields should
    remain the same, except all fields listed in "text_fields" will be removed,
    and two new fields "sentence_text_field" and "sentence_idx_field" will be
    added.
    """
    PipelineOperator.__init__(
      self,
      **kwargs
    )
    self.input_dataset = input_dataset
    self.scispacy_version = scispacy_version
    self.text_fields=text_fields
    self.sentence_text_field = sentence_text_field
    self.sentence_idx_field = sentence_idx_field

  def get_dataframe(self, dask_client:Client)->ddf.DataFrame:
    input_data = dask_client.get_dataset(self.input_dataset)
    new_idx = input_data.columns.drop(
        self.text_fields
    ).append(
        pd.Index([
          self.sentence_text_field,
          self.sentence_idx_field
        ],
        dtype=["object", "int32"])
    )
    meta_df = pd.DataFrame(index=new_idx)
    return input_data.map_partitions(
        text_fields_to_sentences,
        # --
        text_fields=self.text_fields,
        sentence_text_field=self.sentence_text_field,
        sentence_idx_field=self.sentence_idx_field,
        scispacy_version=self.scispacy_version,
        # --
        meta=meta_df,
    ).clear_divisions()

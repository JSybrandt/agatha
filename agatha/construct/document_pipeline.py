import dask
import dask.bag as dbag
from agatha.construct.checkpoint import ckpt
from agatha.construct import (
    construct_config_pb2 as cpb,
    embedding_util,
    ftp_util,
    graph_util,
    semrep_util,
    text_util,
)
from agatha.construct.document_parsers import (
  parse_pubmed_xml,
  parse_covid_json,
)
from agatha.util import misc_util
from pathlib import Path

def get_medline_documents(
    config:cpb.ConstructConfig,
)->dbag.Bag:
  medline_dir = Path(config.medline_xml_dir)
  medline_dir.mkdir(parents=True, exist_ok=True)
  assert medline_dir.is_dir(), f"Failed to make {config.medline_xml_dir}"

  # Download all of pubmed. ####
  if not config.skip_ftp_download:
    print("Downloading pubmed XML Files")
    with ftp_util.ftp_connect(
        address=config.ftp.address,
        workdir=config.ftp.workdir,
    ) as conn:
      # Downloads new files if not already present in shared
      xml_paths = ftp_util.ftp_retreive_all(
          conn=conn,
          pattern="^.*\.xml\.gz$",
          directory=medline_dir,
          show_progress=True,
      )
  else:
    print(f"Skipping FTP download, using {medline_dir}/*.xml.gz instead")
    assert medline_dir.is_dir(), f"Cannot find {medline_dir}"
    xml_paths = list(medline_dir.glob("*.xml.gz"))
    assert len(xml_paths) > 0, f"No .xml.gz files inside {medline_dir}"

  if config.debug.enable:
    print(f"\t- Downsampling {len(xml_paths)} xml files to only "
          f"{config.debug.partition_subset_size}.")
    # Takes the top x (typically larger)
    xml_paths = xml_paths[-config.debug.partition_subset_size:]

  # Parse xml-files per-partition
  medline_documents = dbag.from_delayed([
    dask.delayed(parse_pubmed_xml.parse_zipped_pubmed_xml)(
      xml_path=p,
    )
    for p in xml_paths
  ])

  if not config.allow_nonenglish_abstracts:
    medline_documents = medline_documents.filter(
      # Only take the english ones
      lambda r: r["language"]=="eng"
    )

  if config.HasField("cut_date"):
    # This will fail if the cut-date is not a valid string
    datetime.strptime(config.cut_date, "%Y-%m-%d")
    medline_documents = medline_documents.filter(
        lambda r: r["date"] < config.cut_date
    )

  if config.debug.enable:
    print("\t- Downsampling documents by "
          f"{config.debug.document_sample_rate}")
    medline_documents = medline_documents.random_sample(
        config.debug.document_sample_rate,
    )
  return medline_documents


def get_covid_documents(config:cpb.ConstructConfig)->dbag.Bag:
  covid_json_dir = Path(config.covid_json_dir)
  assert covid_json_dir.is_dir(), \
      f"Failed to find covid_json_dir:{covid_json_dir}"
  json_paths = list(covid_json_dir.glob("**/*.json"))
  assert len(json_paths) > 0, "Failed to find json files in covid_json_dir."
  json_path_bag = dbag.from_sequence(json_paths)
  if config.debug.enable:
    print("\t- Downsampling documents by "
          f"{config.debug.document_sample_rate}")
    json_path_bag = json_path_bag.random_sample(
        config.debug.document_sample_rate,
    )
  json_records = json_path_bag.map(parse_covid_json.json_path_to_record)
  return json_records

def perform_document_independent_tasks(
    config:cpb.ConstructConfig,
    documents:dbag.Bag,
    ckpt_prefix:str,
)->None:
  """
  Performs all of the document processing operations that are required to
  happen on each document separately. This is important to separate between
  different input textual features because this allows us to update/invalidate
  particular sets of checkpoints faster.

  Inputs: All necessary information to process the input set of documents.
  Outputs: Dictionary containing each bag
  """

  ckpt("documents", ckpt_prefix)

  # Split documents into sentences, filter out too-long and too-short sentences.
  sentences = documents.map_partitions(
      text_util.split_sentences,
      # --
      min_sentence_len=config.parser.min_sentence_len,
      max_sentence_len=config.parser.max_sentence_len,
  )
  ckpt("sentences", ckpt_prefix)

  # Get metadata terms from each sentence
  coded_term_edges = graph_util.record_to_bipartite_edges(
    records=sentences,
    get_neighbor_keys_fn=text_util.get_mesh_keys,
    weight_by_tf_idf=False,
  )
  ckpt("coded_term_edges", ckpt_prefix)

  # Make edges between each adj sentence
  adj_sent_edges = graph_util.record_to_bipartite_edges(
    records=sentences,
    get_neighbor_keys_fn=text_util.get_adjacent_sentences,
    # We can store only one side of the connection because each sentence will
    # get their own neighbors. Additionally, these should all have the same
    # sort of connections.
    weight_by_tf_idf=False,
    bidirectional=False,
  )
  ckpt("adj_sent_edges", ckpt_prefix)

  # Apply lemmatization and entity extraction to sentences
  parsed_sentences = sentences.map_partitions(
      text_util.analyze_sentences,
      # --
      text_field="sent_text",
  )
  ckpt("parsed_sentences", ckpt_prefix)

  # Get lemma edges
  lemma_edges = graph_util.record_to_bipartite_edges(
    records=parsed_sentences,
    get_neighbor_keys_fn=text_util.get_interesting_token_keys,
    weight_by_tf_idf=False,
  )
  ckpt("lemma_edges", ckpt_prefix)

  # Get entity edges
  entity_edges = graph_util.record_to_bipartite_edges(
    records=parsed_sentences,
    get_neighbor_keys_fn=text_util.get_entity_keys,
    weight_by_tf_idf=False,
  )
  ckpt("entity_edges", ckpt_prefix)

  # Embed each sentence
  embedded_sentences = (
      sentences
      .map_partitions(
        embedding_util.embed_records,
        # --
        batch_size=config.sys.batch_size,
        text_field="sent_text",
        max_sequence_length=config.parser.max_sequence_length,
      )
  )
  ckpt("embedded_sentences", ckpt_prefix)

  # hash each sentence id
  hashed_embeddings = (
      embedded_sentences
      .map(
        lambda x: {
          "id": misc_util.hash_str_to_int(x["id"]),
          "embedding": x["embedding"]
        }
      )
  )
  ckpt("hashed_embeddings", ckpt_prefix)

  hashed_names = (
      sentences
      .map(lambda rec: {
        "name": rec["id"],
        "hash": misc_util.hash_str_to_int(rec["id"]),
      })
  )
  ckpt("hashed_names", ckpt_prefix)

  # ADD SEMREP FUNCTION CALL HERE


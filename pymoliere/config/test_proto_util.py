# Ensures that the config is loaded and manipuated correctly.

from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from tempfile import TemporaryDirectory
from pathlib import Path
from argparse import Namespace

# Ensures that parsers are working properly
def test_load_serialized_pb_as_proto():
  expected = cpb.FtpSource()
  expected.address = "abcd"
  with TemporaryDirectory() as tmp_dir:
    path = Path(tmp_dir).joinpath("ftp_source.pb")
    with open(path, 'wb') as tmp_file:
      tmp_file.write(expected.SerializeToString())
    actual = proto_util.load_serialized_pb_as_proto(
      path=path,
      proto_obj=cpb.FtpSource()
    )
  assert actual == expected

def test_load_json_as_proto():
  expected = cpb.FtpSource()
  expected.address = "abcd"
  with TemporaryDirectory() as tmp_dir:
    path = Path(tmp_dir).joinpath("ftp_source.json")
    with open(path, 'w') as tmp_file:
      tmp_file.write("""
      {
        "address": "abcd"
      }
      """)
    actual = proto_util.load_json_as_proto(
      path=path,
      proto_obj=cpb.FtpSource()
    )
  assert actual == expected

def test_load_text_as_proto():
  expected = cpb.FtpSource()
  expected.address = "abcd"
  with TemporaryDirectory() as tmp_dir:
    path = Path(tmp_dir).joinpath("ftp_source.config")
    with open(path, 'w') as tmp_file:
      tmp_file.write("""
        address: "abcd"
      """)
    actual = proto_util.load_text_as_proto(
      path=path,
      proto_obj=cpb.FtpSource()
    )
  assert actual == expected

def test_parse_proto_fields_ftp_source():
  expected = set(["address", "dir_path"])
  actual = set(proto_util.get_full_field_names(cpb.FtpSource()))
  assert actual == expected

def test_parse_proto_fields_build_config():
  expected = set([
    "cluster.head_address",
    "cluster.port",
    "dataserver.address",
    "ftp_source.address",
    "ftp_source.dir_path",
  ])
  actual = set(proto_util.get_full_field_names(cpb.ConstructConfig()))
  # Assert that the ConstructConfig has at least these names
  assert actual.intersection(expected) == expected

def test_setup_parser_with_proto():
  parser = proto_util.setup_parser_with_proto(cpb.ConstructConfig())
  args = parser.parse_args([])
  assert hasattr(args, "cluster.head_address")
  assert hasattr(args, "cluster.port")
  assert hasattr(args, "dataserver.address")
  assert hasattr(args, "ftp_source.address")
  assert hasattr(args, "ftp_source.dir_path")

def test_set_field_nested():
  expected = cpb.ConstructConfig()
  expected.cluster.head_address = "new_addr_val"
  actual = cpb.ConstructConfig()
  proto_util.set_field(actual, "cluster.head_address", "new_addr_val")
  assert actual == expected

def test_set_field_unnested():
  expected = cpb.FtpSource()
  expected.address = "new_addr_val"
  actual = cpb.FtpSource()
  proto_util.set_field(actual, "address", "new_addr_val")
  assert actual == expected

def test_transfer_args_to_proto():
  actual = cpb.ConstructConfig()
  actual.cluster.head_address = "original_addr_val"
  actual.cluster.port = 1234
  actual.ftp_source.address = "unrelated"
  # Overwrite some values with ns
  ns = Namespace()
  setattr(ns, "cluster.head_address", "NEW_addr_val")
  setattr(ns, "cluster.port", 4321)
  proto_util.transfer_args_to_proto(ns, actual)

  expected = cpb.ConstructConfig()
  expected.cluster.head_address = "NEW_addr_val"
  expected.cluster.port = 4321
  expected.ftp_source.address = "unrelated"

  assert actual == expected

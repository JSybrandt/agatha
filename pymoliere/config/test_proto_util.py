# Ensures that the config is loaded and manipuated correctly.

from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
from tempfile import TemporaryDirectory
from pathlib import Path

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

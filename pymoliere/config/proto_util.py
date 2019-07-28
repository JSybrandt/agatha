# This module is responsible for parsing the config objects and modifying them with command line args.

from . import config_pb2
from google.protobuf import (
  json_format,
  text_format,
)
from pathlib import Path
from argparse import ArgumentParser
from typing import (
    TypeVar,
    Callable,
    Dict,
)

# Generic proto class
ProtoObj = TypeVar('ProtoObj')

# Function template for proto parsing functions.
# Each reads a file and produces a requested proto.
ProtoLoadFn = Callable[[Path, ProtoObj], ProtoObj]

def load_serialized_pb_as_proto(
  path:Path,
  proto_obj: ProtoObj
) -> ProtoObj:
  """
  Interprets the file `path` as a serialized proto. Reads data into `proto_obj`.
  Returns a reference to the same `proto_obj`
  """
  assert path.is_file()
  with path.open('rb') as p_file:
    proto_obj.ParseFromString(p_file.read())
  return proto_obj

def load_json_as_proto(
  path: Path,
  proto_obj: ProtoObj
) -> ProtoObj:
  """
  Interprets `path` as a plaintext json file. Reads the data into `proto_obj`
  and returns a reference with the result.
  """
  assert path.is_file()
  with path.open('r') as j_file:
    json_format.Parse(j_file.read(), proto_obj)
  return proto_obj

def load_text_as_proto(
  path: Path,
  proto_obj: ProtoObj
) -> ProtoObj:
  """
  Interprets `path` as a plaintext proto file. Reads the data into `proto_obj`
  and returns a reference with the result.
  """
  assert path.is_file()
  with path.open('r') as t_file:
    text_format.Merge(t_file.read(), proto_obj)
  return proto_obj

EXT_TO_PROTO_PARSER: Dict[str, ProtoLoadFn] = {
  ".pb": load_serialized_pb_as_proto,
  ".json": load_json_as_proto,
  ".txt": load_text_as_proto,
  ".config": load_text_as_proto,
}

def read_file_as_proto(
  path: Path,
  proto_obj: ProtoObj
) -> ProtoObj:
  """
  Attempts to parse the provided file using all available parsers.
  Raises exception if unavailable.
  """
  assert path.is_file()
  if path.suffix in EXT_TO_PROTO_PARSER:
    try:
      return EXT_TO_PROTO_PARSER[path.suffix](path, proto_obj)
    except:
      raise SyntaxError(f"Unable to parse {path}.")
  else:
    raise NotImplementedError(f"Unsupported config file type: {path.suffix}.")

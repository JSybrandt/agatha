"""
This module contains the functions necessary to load and manipulate proto
objects. Key components of this module include the load_proto function, that
wraps multiple proto parsers, as well as parse_args_into_proto, which loads and
augments a proto from the command line.
"""

from pymoliere.config import config_pb2 as cpb
from google.protobuf import (
  json_format,
  text_format,
)
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Dict,
    List,
    TypeVar,
)
from argparse import ArgumentParser, Namespace

# Generic proto class
ProtoObj = TypeVar('ProtoObj')

# Function template for proto parsing functions.
# Each reads a file and produces a requested proto.
ProtoLoadFn = Callable[[Path, ProtoObj], ProtoObj]

################################################################################

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

def load_proto(
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

################################################################################

def get_field(proto_obj:ProtoObj, field:str) -> Any:
  """
  Looks up a message or field from the supplied proto in the same way that
  getattr might. However, this function is smart enough to handle nested
  messages in terms of "a.b.c"
  """
  result = proto_obj
  for name in field.split("."):
    result = getattr(result, name)
  return result

def set_field(proto_obj:ProtoObj, field:str, val:Any) -> None:
  tokens = field.split(".")
  if len(tokens) > 1:
    proto_obj = get_field(proto_obj, ".".join(tokens[:-1]))
  setattr(proto_obj, tokens[-1], val)

def get_full_field_names(proto_obj: ProtoObj)->List[str]:
  """
  Lists all field names in all nested messages in the given proto.
  For instance:
  :  message Foo {
  :    optional string str_1 = 1;
  :    optional string str_2 = 2;
  :  }
  :  message Bar {
  :    optional Foo foo = 1;
  :    optional int32 num = 2;
  :  }
  get_full_field_names(Foo()) contains [str_1, str_2]
  get_full_field_names(Bar()) contains [num, foo.str_1, foo.str_2]
  """
  def get_message_names(obj):
    return [f.name for f in obj.DESCRIPTOR.fields]
  def is_message(obj):
    return hasattr(obj, "DESCRIPTOR")

  # Start with all names in the top-level message
  name_stk = get_message_names(proto_obj)
  non_message_fields = []
  # Doing a dfs through the message
  while len(name_stk) > 0:
    curr_name = name_stk.pop()
    curr_obj = get_field(proto_obj, curr_name)
    # If the current message contains nested children
    if is_message(curr_obj):
      name_stk += [
        f"{curr_name}.{child_name}"
        for child_name
        in get_message_names(curr_obj)
      ]
    else:
      non_message_fields.append(curr_name)
  return non_message_fields

def setup_parser_with_proto(config_proto:ProtoObj)->ArgumentParser:
  """
  This class parses command line arguments into the config_proto.
  The idea is to have sensible defaults at all levels.
  Default Levels:
    - Written in proto definition
    - Written in config file
    - Written on command line
  """
  parser = ArgumentParser(
    description="Build and Query a PyMoliere Network."
  )
  parser.add_argument(
    "config",
    type=Path,
    nargs="?",
    help="If supplied, load the given config proto. Note, command-line "
         "options will override values written in this config file.",
  )
  for field in get_full_field_names(config_proto):
    val = get_field(config_proto, field)
    parser.add_argument(
        f"--{field}",
        type=type(val),
        default=val,
    )
  return parser

def transfer_args_to_proto(args:Namespace, config_proto:ProtoObj) -> ProtoObj:
  """
  Writes the fields from the args to the config proto.
  """
  for field in get_full_field_names(config_proto):
    if hasattr(args, field):
      set_field(config_proto, field, getattr(args, field))
  return config_proto

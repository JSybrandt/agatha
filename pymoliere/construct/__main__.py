from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
if __name__ == "__main__":
  config = cpb.BuildConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.transfer_args_to_proto(
      proto_util.setup_parser_with_proto(config).parse_args(),
      config
  )
  print("Running pymoliere build with the following parameters:")
  print(config)

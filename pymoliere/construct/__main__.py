from pymoliere.config import (
    config_pb2 as cpb,
    proto_util,
)
if __name__ == "__main__":
  config = cpb.ConstructConfig()
  # Creates a parser with arguments corresponding to all of the provided fields.
  # Copy any command-line specified args to the config
  proto_util.parse_args_to_config_proto(config)
  print("Running pymoliere build with the following parameters:")
  print(config)
  print(config.ftp_source.address)

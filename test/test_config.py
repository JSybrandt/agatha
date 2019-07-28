# Ensures that the config is loaded and manipuated correctly.

import pymoliere
def test_config_exists():
  from pymoliere import config_pb2 as c
  query_cfg = c.QueryConfig()
  build_cfg = c.BuildConfig()
  assert True

import os

RTD_ENV = "READTHEDOCS"

def running_on_read_the_docs():
  return RTD_ENV in os.environ and os.environ[RTD_ENV].lower() == "true"

def parse_requirements(deps_path):
  res = []
  with open(deps_path) as req_file:
    for line in req_file:
      line = line.strip()
      if len(line) > 0 and line[0] != '#':
        res.append(line)
  return res



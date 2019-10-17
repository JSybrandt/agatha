def _get_git_head():
  import os
  import subprocess
  curr_path = os.path.dirname(os.path.abspath(__file__))
  return subprocess.check_output([
    "git",
    "-C",
    curr_path,
    "log",
    "--oneline",
    "-n",
    "1",
    "HEAD",
  ]).strip().decode()

__VERSION__ = '19.10.1',
__GIT_HEAD__ = _get_git_head()

from agatha.construct import file_util
from pathlib import Path
from multiprocessing import Process
import time
import pytest


def _wait_and_touch(file_path:Path, wait_before:float)->None:
  time.sleep(wait_before)
  file_path = Path(file_path)
  file_path.touch()

def async_touch_after(file_path:Path, wait_before:float)->None:
  """
  Spawns a process to touch a file
  """
  file_path = Path(file_path)
  if file_path.is_file():
    file_path.unlink()
  Process(target=_wait_and_touch, args=(file_path, wait_before)).start()

def test_async_touch_after():
  path = Path("/tmp").joinpath("test_async_touch_after")
  async_touch_after(path, .5)
  assert not path.is_file()
  time.sleep(2)
  assert path.is_file()

def test_wait_for_file_to_appear_exists():
  path = Path("/tmp").joinpath("test_wait_for_file_to_appear_exists")
  async_touch_after(path, .5)
  file_util.wait_for_file_to_appear(path, max_tries=2)

def test_wait_for_file_to_appear_not_exists():
  path = Path("/tmp").joinpath("test_wait_for_file_to_appear_not_exists")
  with pytest.raises(AssertionError):
    file_util.wait_for_file_to_appear(path, max_tries=2)


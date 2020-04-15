from agatha.construct import checkpoint
from pathlib import Path
from agatha.construct.file_util import DONE_FILE


def test_set_root():
  checkpoint._reset_param()
  expected = Path()
  checkpoint.set_root(expected)
  assert checkpoint.get_root() == expected

def test_default_none_ckpt_root():
  checkpoint._reset_param()
  assert checkpoint._PARAM["ckpt_root"] is None

def test_get_or_make_ckpt_dir():
  checkpoint._reset_param()
  root = Path("/tmp")
  name = "test_get_or_make_ckpt_dir"
  expected = root.joinpath(name)

  checkpoint.set_root(root)
  actual = checkpoint.get_or_make_ckpt_dir(name)
  assert actual == expected
  assert expected.is_dir()

def test_clear_ckpt():
  checkpoint._reset_param()
  root = Path("/tmp")
  name = "test_clear_ckpt"
  expected = root.joinpath(name)

  checkpoint.set_root(root)
  actual = checkpoint.get_or_make_ckpt_dir(name)
  assert actual == expected
  assert expected.is_dir()

  checkpoint.clear_ckpt(name)
  assert not expected.exists()

def setup_done_checkpoints(root_name:str)->None:
  checkpoint._reset_param()
  root = Path("/tmp/"+root_name)
  checkpoint.set_root(root)
  checkpoint.clear_all_ckpt()
  all_checkpoints = [
      "1_a", "2_a", "3_a", "1_b", "2_b", "3_b", "1_c", "2_c", "3_c",
  ]
  for ckpt_name in all_checkpoints:
    ckpt_dir = checkpoint.get_or_make_ckpt_dir(ckpt_name)
    print("Making:", ckpt_dir)
    # Set the ckpt to complete
    done_file = ckpt_dir.joinpath(DONE_FILE)
    with open(done_file, "w"):
      print("Making", done_file)

  return all_checkpoints

def test_setup_done_checkpoints():
  for ckpt_name in setup_done_checkpoints("test_setup_done_checkpoints"):
    assert checkpoint.is_ckpt_done(ckpt_name), f"{ckpt_name} is not DONE"

def test_get_checkpoints_like():
  setup_done_checkpoints("test_get_checkpoints_like")
  expected = { "1_a", "2_a", "3_a", }
  actual = checkpoint.get_checkpoints_like("*_a")
  assert expected == actual

def test_get_checkpoints_like_complete():
  setup_done_checkpoints("test_get_checkpoints_like_complete")

  # clear 2_a
  checkpoint.clear_ckpt("2_a")
  checkpoint.get_or_make_ckpt_dir("2_a")

  # 2_a is not DONE
  expected = { "1_a", "3_a" }
  actual = checkpoint.get_checkpoints_like("*_a")

  assert expected == actual

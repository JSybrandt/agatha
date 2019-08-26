from ftplib import FTP
from typing import List
from pathlib import Path
import re
from tqdm import tqdm

def ftp_connect(address:str, workdir:str)-> FTP:
  "Connects to a remote FTP server at a specific directory"
  conn = FTP(address)
  conn.login()
  conn.cwd(workdir)
  return conn


def ftp_list_files(conn:FTP, pattern:str=".*") -> List[str]:
  pattern = re.compile(pattern)
  return [f for f in conn.nlst() if pattern.match(f)]


def ftp_download(conn:FTP, remote_name:str, local_dir:Path) -> Path:
  local_path = local_dir.joinpath(remote_name)
  assert not local_path.exists()
  with local_path.open('wb') as local_file:
    conn.retrbinary(f"RETR {remote_name}", local_file.write, 1024)
  return local_path


def ftp_download_if_missing(conn:FTP, remote_name:str, local_dir:Path) -> Path:
  "If the file already exists, skip it."
  assert local_dir.is_dir()
  local_path = local_dir.joinpath(remote_name)
  if local_path.is_file():
    return local_path
  else:
    assert not local_path.exists()
    return ftp_download(
        conn=conn,
        remote_name=remote_name,
        local_dir=local_dir
    )


def ftp_retreive_all(
  conn:FTP,
  local_dir:Path,
  pattern:str=".*",
  show_progress:bool=False,
) -> List[Path]:
  """
  For each file matching the given pattern, download if not in local_dir.
  """
  assert local_dir.is_dir()
  files = ftp_list_files(
    conn=conn,
    pattern=pattern,
  )
  res = []
  for f in tqdm(files, disable=(not show_progress)):
    res.append(
      ftp_download_if_missing(
        conn=conn,
        remote_name=f,
        local_dir=local_dir
      )
    )
  return res

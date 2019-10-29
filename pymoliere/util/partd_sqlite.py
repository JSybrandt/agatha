from __future__ import absolute_import

from partd.core import Interface
from partd.utils import ignoring
from sqlitedict import SqliteDict
import locket

class SqliteInterface(Interface):
  def __init__(self, path=None, dir=None):
    if path == None:
      path = tempfile.mkdtemp(suffix='.partd.sql', dir=dir)
    self.db = SqliteDict(path)
    self.path = path
    self.lock = locket.lock_file(path+'.lock')

  def __del__(self):
    self.db.close()
    if lock: self.lock.release()


  def _get(self, keys, lock=True, **kwargs):
    assert isinstance(keys, (list, tuple, set))
    if lock:
      self.lock.acquire()
    try:
      result = []
      for key in keys:
        try:
          result.append(self.db[key])
        except KeyError:
          result.append(b'')
    finally:
      if lock:
        self.lock.release()
    return result

  def _iset(self, key, value, lock=True):
    """ Idempotent set """
    if lock:
      self.lock.acquire()
    try:
      self.db[key] = value
    finally:
      if lock:
        self.lock.release()
      self.db.commit()


  def _delete(self, keys, lock=True):
    if lock:
      self.lock.acquire()
    try:
      for key in keys:
        del self.db[key]
    finally:
      if lock:
        self.lock.release()
      self.db.commit()

  def append(self, data, lock=True, fsync=False, **kwargs):
    if lock:
      self.lock.acquire()
    try:
      for k, v in data.items():
        if k not in self.db:
          self.db[k] = b""
        self.db[k] += v
    finally:
      if lock:
        self.lock.release()
      self.db.commit()

  def drop(self):
    self.db.close()

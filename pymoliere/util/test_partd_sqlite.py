
from partd import Pickle
from pymoliere.util.partd_sqlite import SqliteInterface
import os

def test_sqlite_partd():
    path = 'tmp.partd.sql'
    with SqliteInterface(path=path) as p:
      assert os.path.exists(path)
      p.append({'x': b'Hello!'})
      p.append({'y': b'World!'})
      assert p.get('x') == b'Hello!'
      assert p.get('y') == b'World!'

      assert p.get(['y', 'x']) == [b'World!', b'Hello!']
      p.append({'x': b'Again!'})
      assert p.get('x') == b'Hello!Again!'

      assert p.get('z') == b''
    os.unlink(path)

def test_sqlite_with_pickle():
  path = 'tmp.partd.sql'
  with Pickle(SqliteInterface(path)) as p:
    assert os.path.exists(path)
    p.append({"x": ["Hello!"]})
    p.append({"x": ["World!"]})
    obj = {str(x): str(x**2) for x in range(10)}
    p.append({"y": [obj]})
    p.append({'z': [1,2,3]})
    p.append({'z': [4,5,6]})

    assert p.get('x') == ["Hello!", "World!"]
    assert p.get('y') == [obj]
    assert p.get('z') == list(range(1,7))
    assert p.get('not_found') == []

  os.unlink(path)

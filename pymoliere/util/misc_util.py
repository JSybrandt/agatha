from itertools import zip_longest

def iter_to_batches(iterable, batch_size):
  args = [iter(iterable)] * batch_size
  for batch in zip_longest(*args):
    yield list(filter(lambda b: b is not None, batch))

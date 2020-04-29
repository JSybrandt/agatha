How to Train Agatha
===================

Training the Agatha deep learning model is the last step to generating
hypotheses after you've already processed all necessary information using
`agatha.construct`. This process uses [PyTorch][1] and [PyTorch-Lightning][2] to
efficiently manage the distributed training of the predicate ranking model
stored in `agatha.ml.hypothesis_predictor`.

## Background

The Agatha deep learning model learns to rank entity-pairs. To learn this
ranking, we will be comparing existing predicates found within our dataset
against randomly sampled entity-pairs. Of course, if a predicate exists in our
database, it should receive a higher model output than many random pairs.

A `positive sample` is a entity-pair that actually occurs in our dataset. A
`negative sample` is one of those non-existent randomly sampled pairs. We will
use the [margin ranking loss][3] criteria to learn to associate higher values
with positive samples. To do this, we will compare one positive sample to a high
number of negative samples. This is the `negative-sampling rate`.

A single sample, be it positive or negative, is comprised of four parts:

 1. Term 1 (the subject).
 2. Term 2 (the object).
 3. Predicates associated with term 1 (but not term 2).
 4. Predicates associated with term 2 (but not term 1).

This as a whole is referred to as a `sample`. Generating samples is the primary
bottleneck in the training process. This is because we have many millions of
terms and predicates. As a result, the Agatha deep learning framework comes
along with a number of utilities to make managing the large datasets easier.

## Datasets

In order to begin training you will need the following data:

 1. Embeddings for all entities and predicates, stored as a directory of `.h5`
    files.
 2. Entity metadata, stored as a `.sqlite3` file.
 2. The subgraph containing all entity-predicate edges, stored as a `.sqlite3`
    file.

The network construction process will produce these datasets as `sqlite3` files.
[Sqlite][4] is an embedded database, meaning that we can load the database from
storage and don't need to spin up a whole server. Additionally, because we are
only going to _read_ and never going to _write_ to these databases during
training, each machine in our distributed training cluster can have independent
access to the same data very efficiently.

All of the sqlite3 databases managed by Agatha are stored in a simple format
that enables easy python access through the
`agatha.util.sqlite3_lookup.Sqlite3LookupTable` object. This provides read-only
access to the database through a dictionary-like interface.

For instance, if we want to get the neighbors for the node `l:noun:cancer`, we
can simply write this code:

```python3
from agatha.util.sqlite3_lookup import Sqlite3LookupTable
graph = Sqlite3LookupTable("./data./releases/2020/graph.sqlite3")
graph["l:noun:cancer"]
# Returns:
# ... [
# ...   < List of all neighbors >
# ... ]
```

This process works by first making an sqlite3 connection to the graph database
file. By default, we assume that this database contains a table called
`lookup_table` that has the schema: `(key:str, value:str)`. Values in this
database are all json-encoded. This means that calling `graph[foo]`  looks up
the value associated with "foo" in the database, and parses whatever it find
through `json.loads(...)`.

This process is slow compared to most other operations in the training pipeline.
Each query has to check against the sqlite `key` index, which is stored on disk,
load the `value`, also stored on disk, and then parse the string. There are two
optimizations that make this faster: preloading and caching. Look into the API
documentation for more detail


[1]:https://pytorch.org/
[2]:https://github.com/PytorchLightning/pytorch-lightning
[3]:https://pytorch.org/docs/stable/nn.html#torch.nn.MarginRankingLoss
[4]:https://www.sqlite.org/index.html

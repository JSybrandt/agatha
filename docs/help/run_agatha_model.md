How to Run Your Agatha Model
============================

The first set to run your agatha model is to either train your own, or download
a pretrained model. Other help pages describe how to do this in more detail.

To begin, lets assume you downloaded [Agatha 2020][agatha_2020]. When extracted,
you should see the following directory structure. This is consistent with other
pretrained models.

```
model_release/
  model.pt
  predicate_embeddings/
    embeddings_*.v5.h5
  predicate_entities.sqlite3
  predicate_graph.sqlite3
```

## TL;DR

Once you have the necessary files, you can run the model with the following
snippet:

```python3
# Example paths, change according to your download location
model_path = "model_release/model.pt"
embedding_dir = "model_release/predicate_embeddings"
entity_db_path = "model_release/predicate_entities.sqlite3"
graph_db_path = "model_release/predicate_graph.sqlite3"

# Load the model
import torch
model = torch.load(model_path)

# Configure auxilary data paths
model.configure_paths(
  embedding_dir=embedding_dir
  entity_db=entity_db_path,
  graph_db=graph_db_path,
)

# Now you're ready to run some predictions!
# C0006826 : Cancer
# C0040329 : Tobacco
model.predict_from_terms([("C0006826", "C0040329")])
>>> [0.9946276545524597]

# Speed up by moving the model to GPU
model = model.cuda()

# If we would like to run thousands of queries, we want to load everything
# before the query process. This takes a while, and is optional.
model.preload()

# Get a list of valid terms (slow if preload not called beforehand)
from agatha.util.entity_types import is_umls_term_type
valid_terms = list(filter(is_umls_term_type, model.graph.keys()))
len(valid_terms)
>>> 278139

# Run a big batch of queries
from random import choice
rand_term_pairs =  [
 (choice(valid_terms), choice(valid_terms))
 for _ in range(100)
]
scores = model.predict_from_terms(rand_term_pairs, batch_size=64)
# Now, scores[i] corresponds to rand_term_pairs[i]

```

## Required Files

`model.pt`
: This is the model file. You can load this with `torch.load`. Behind the
  scenes, this will startup the appropriate Agatha module.

`predicate_embeddings`
: This directory contains graph embeddings for each entity needed to make
  predicate predictions using the Agatha model.

`predicate_entities.sqlite3`
: This database contains embedding metadata for each entity managed by the
  Agatha model. This database is loaded with
  `agatha.util.sqlite3_lookup.Sqlite3LookupTable`.

`predicate_graph.sqlite3`
: This database contains term-predicate relationships for each entity managed by
  the Agatha model. This database is loaded with
  `agatha.util.sqlite3_lookup.Sqlite3LookupTable`.

## Bulk queries

In order to run bulk queries efficiently, you will want to run:

```python3
model.cuda()
model.preload()
```

The first command, `model.cuda()` moves the weight matrices to the GPU. The
second command, `model.preload()` modes all graph and embedding information into
RAM. This way, each request for an embedding, of which we will do tens of times
per query, can be handled without a slow lookup in the storage system. Warning,
expect this to take around 30 GB of RAM to start. Additionally, Agatha employs
caching intermediate values that will increase the memory usage as the query
process goes on.

### Batch size

When running `model.predict_from_terms` the optional `batch_size` parameter can
be used to improve GPU usage. Set this value to an integer greater than one to
pack more than one query within each call to the GPU. You may need to
experiment to find a value that is large, but doesn't exceed GPU memory.


[agatha_2020]:https://drive.google.com/open?id=1GLKh9OJI0QVfeDZga2XlnMTa8bQGhp1F

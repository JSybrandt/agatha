Topic Model Queries on the Agatha Semantic Network
==================================================

Our prior work, [Moliere][moliere], performed hypothesis generation through a
graph-analytic and topic-modeling approach. Occasionally, we would like to run
this same approach using the Agatha topic network. This document describes the
way to use the `agatha.topic_query` module to perform topic-model queries, and
how to interpret your results.

## TL;DR

This is the recommended way to run the query process. First, create a file
called `query.conf` and fill it with the following information:

```proto
graph_db: "<path to graph.sqlite3>"
bow_db: "<path to sentences.sqlite3>"
topic_model {
  num_topics: 100
}
```

Look into `agatha/topic_query/topic_query_config.proto` to get more details on
the `TopicQueryConfig` specification.

Now you can run queries using the following syntax:

```bash
python3 -m agatha.topic_query query.conf \
  --source <source term> \
  --target <target term> \
  --result_path <desired place to put result>
```

Here is a real-life example of a query:

```bash
python3 -m agatha.topic_query configs/query_2020.conf \
  --source l:noun:tobacco \
  --target l:noun:cancer \
  --result_path ./tobacco_cancer.pb
```

## Viewing Results

Once you're done your query, you will have a binary file containing all topic
model information. This is stored as a compressed proto format, which should
enable easy programmatic access to all the components of the query result. You
can view more details on the proto specification at
`agatha/query/topic_query_result.proto`.

Here's a short python script that would load a proto result file for use:

```python3
from agatha.topic_query import topic_query_result_pb2
result = topic_query_result_pb2.TopicQueryResult()
with open("<result path>", 'rb') as proto_file:
  result.ParseFromString(proto_file.read())
```

You now have access to: `result.path`, `result.documents`, and `result.topics`.

If you want to cut to the chase, you can simply print out all proto result
details using the following script:

## Running Queries with Node Names

In order to run queries, you will need to know the particular node names of the
elements you would like to explore. Nodes of the Agatha network can be explored
by looking at the set of `node` entities in the graph database. You can explore
these in `sqlite3` with the following syntax:

```bash
sqlite3 .../graph.sqlite3 \
  'select node from graph where node like "%<query term>%" limit 10'
```

Here's an actual example:

```bash
sqlite3 graph.sqlite3 'select node from graph where node like "%dimentia%" limit 10'
  > e:amyotrophic_lateral_sclerosis/parkinsonism_dimentia_complex
  > e:dimentia_complex
  > e:hiv-associated_dimentia
  > e:mild_dimentia
  > e:three-dimentianl_(_3d_)
  > l:adj:three-dimentianl
  > l:noun:dimentia
```

Note that node names follow particular patterns. All valid node names start with
a leading "type" character. These are specified in
`agatha/util/entity_types.py`. Here are the existing entity types at the time of
writing:

```python3
ENTITY_TYPE="e"
EMMA_TYPE="l"
MESH_TERM_TYPE="m"
UMLS_TERM_TYPE="m"
NGRAM_TYPE="n"
PREDICATE_TYPE="p"
SENTENCE_TYPE="s"
```

## Configuration

Just like the Agatha network construction process, the query process also needs
many parameters that are specified either through command-line arguments, or
through a configuration script. We recommend creating a configuration for the
typical query case, omitting only the query term parameters. This way you can
have the simplest query interface when running these topic-model queries
yourself.

Look into `agatha/config/topic_query_config.proto` to get more details on the
`TopicQueryConfig` specification. Here is a fuller example of a configuration
that we actually use on Palmetto.

```proto
# TopicQueryConfig

# source: Omitted
# target: Omitted
# result_path: Omitted

graph_db: "/zfs/safrolab/users/jsybran/agatha/data/releases/2020/graph.sqlite3"
bow_db: "/zfs/safrolab/users/jsybran/agatha/data/releases/2020/sentences.sqlite3"
topic_model {
  num_topics: 20
  min_support_count: 2
  truncate_size: 250
}

# Advanced

max_sentences_per_path_elem: 2000
max_degree: 1000
```

[moliere]:https://github.com/JSybrandt/MOLIERE

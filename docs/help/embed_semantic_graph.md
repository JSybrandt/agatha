How to Embed the Agatha Semantic Graph
======================================

We use [Pytorch Big Graph
(PTBG)](https://torchbiggraph.readthedocs.io/en/latest/) to embed our semantic
graph. This is a distributed knowledge graph embedding tool, meaning that it
uses multiple machines, and takes node / edge type into account when embedding.
PTBG is a complex tool that requires a number of preprocessing steps to use.

## PTBG Process Outline

 1. Create a single directory that contains all semantic graph edges.
   - This is produced by running `agatha.construct`.
   - Edges are stored as small key-value json files.
   - The directory may contain a large number of files.
 2. Convert graph to PTBG input format.
   - PTBG requires that we index and partition all nodes and edges.
   - Look into `tools/convert_graph_for_pytorch_biggraph` for how to do this.
 3. Create a PTBG config.
   - Specify all node / edge types.
   - Specify location of all input files.
   - The parameters of this config must match the options used in the
     conversion.
 4. Launch the PTBG training cluster.
   - Use 10-20 machines, too many will slow this process.
   - Wait at least 5 epochs, will take days.
 5. Index the resulting embeddings for use in Agatha.
   - Agatha needs to know where to find each embedding, given the node name.
   - Use `tools/py_scripts/ptbg_index_embeddings.py` to create a lookup table
     that maps each node name to its embedding metadata.

## Convert Edges to PTBG format

The PTBG conversion tool is a multi-threaded single-machine program that indexes
every node and edge of the input graph for PTBG distributed training. The
settings used to run this tool will determine qualities of the resulting PTBG
config, so you will want to save the exact command you run for later steps.

**WARNING:** This program is extremely memory intensive. If you're running on
plametto, make sure to grab the 1.5 or 2 TB node.

To begin, build the `convert_graph_for_pytorch_biggraph` tool.

```bash
cd /path/to/agatha/tools/convert_graph_for_pytorch_biggraph
make
```

This will produce `graph_to_ptbg`. You can take a look at how this tool works
with the `./graph_to_ptbg --help` command.

If you want to embed the entire graph, you can run this conversion with:

```bash
./graph_to_ptbg -i <json_edge_dir> -o <ptbg_data_dir>
```

By default, this will include all expected node and relationship types, as
described in the Agatha paper.

If you only want to embed part of the graph, you can select the specific node
and relation types to include. Note that excluded types will be ignored.

To select as subset of nodes, you will need to supply the optional `--types` and
`--relations` arguments. Here's an example of using these flags to select only
nodes and relationships between umls terms (type m) and predicates (type p).

```bash
./graph_to_ptbg \
  -i <json_edge_dir> \
  -o <ptbg_data_dir> \
  --types "mp" \
  --relations "mp pm"
```

Note that the argument passed with `--types` should be a string where each
character indicates a desired node type. Nodes of types outside of this list
will not be included in the output.

Note that the argument passed with `--relations` should be a string with
space-separated relationship types. Each relationship should be a two character
long string. Relationships are also directed in PTBG, meaning that if you would
like to select both `UMLS -> predicate` edges, as well as `predicate -> UMLS`
edges, you will need to specify both edge types.

**WARNING:** You will need to remember the order you list the relationships.
This will determine the order of relationships in the PTBG config.

## Create a PTBG Config

Now that you have converted the agatha semantic graph for PTBG, you now need to
write a configuration script. Here's the [official docs for the PTBG
config](https://torchbiggraph.readthedocs.io/en/latest/configuration_file.html).
The following is an example PTBG config. The parts you need to worry about occur
in the header section of the `get_torchbiggraph_config` function. You should
copy this and change what you need.

```python3
#!/usr/bin/env python3
def get_torchbiggraph_config():

    # CHANGE THESE #########################################################

    DATA_ROOT = "/path/to/data/root"
    """ This is the location you specified with the `-o` flag when running
    `convert_graph_for_pytorch_biggraph` That tools should have created
    `DATA_ROOT/entities` and `DATA_ROOT/edges`. This process will create
    `DATA_ROOT/embeddings`.  """

    PARTS = 100
    """ This is the number of partitions that all nodes and edges have been
    split between when running `convert_graph_for_pytorch_biggraph`. By default,
    we create 100 partitions. If you specified `--partition-count` (`-c`), then
    you need to change this value to reflect the new partition count.  """

    ENT_TYPES = "selmnp"
    """ This is the set of entities specified when running
    `convert_graph_for_pytorch_biggraph`. The above value is the default. If you
    used the `--types` flag, then you need to set this value accordingly."""

    RELATIONS = [ "ss", "se", "es", "sl", "ls", "sm", "ms", "sn", "ns", "sp",
                  "ps", "pn", "np", "pm", "mp", "pl", "lp", "pe", "ep" ]
    """ This is the ordered list of relationships that you specified when
    running `convert_graph_for_pytorch_biggraph`. The above is the default. If
    you specified `--relations` then you need to set this value accordingly.
    WARNING: The order of relationships matters! This list should be in the same
    order as the relationships specified in the `--relations` argument.
    """

    EMBEDDING_DIM = 512
    """ This is the number of floats per embedding per node in the resulting
    embedding. """
    
    NUM_COMPUTE_NODES = 20
    """ This is the number of computers used to compute the embedding. We find
    that around 20 machines is the sweet spot. More or less result in slower
    embeddings. """

    THREADS_PER_NODE = 24
    """ This is the number of threads that each machine will use to compute
    embeddings. """

    #########################################################################

    config = dict(
        # IO Paths
        entity_path=DATA_ROOT+"/entities",
        edge_paths=[DATA_ROOT+"/edges"],
        checkpoint_path=DATA_ROOT+"/embeddings",

        # Graph structure
        entities={t: {'num_partitions': PARTS} for t in ENT_TYPES},
        relations=[
          dict(name=rel, lhs=rel[0], rhs=rel[1], operator='translation')
          for rel in RELATIONS
        ],

        # Scoring model
        dimension=EMBEDDING_DIM,
        comparator='dot',
        bias=True,

        # Training
        num_epochs=5,
        num_uniform_negs=50,
        loss_fn='softmax',
        lr=0.02,

        # Evaluation during training
        eval_fraction=0,

        # One per allowed thread
        workers=THREADS_PER_NODE,
        num_machines=NUM_COMPUTE_NODES,
        distributed_init_method="env://",
        num_partition_servers=-1,
    )

    return config
```

## Launch the PTBG training cluster

Now you are ready to start training!

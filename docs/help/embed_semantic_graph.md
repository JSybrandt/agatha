How to Embed the Agatha Semantic Graph
======================================

We use [Pytorch Big Graph
(PTBG)][https://torchbiggraph.readthedocs.io/en/latest/] to embed our semantic
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
like to select both UMLS -> predicate edges, as well as predicate -> UMLS edges,
you will need to specify both edge types.

**WARNING:** You will need to remember the order you list the relationships.
This will determine the order of relationships in the PTBG config.

How to Embed the Agatha Semantic Graph
======================================

We use [Pytorch Big Graph
(PTBG)][https://torchbiggraph.readthedocs.io/en/latest/] to embed our semantic
graph. This is a distributed knowledge graph embedding tool, meaning that it
uses multiple machines, and takes node / edge type into account when embedding.
PTBG is a complex tool that requires a number of preprocessing steps to use.

## PTBG Process Outline

 1. Create a single directory that contains all semantic graph edges.
   - Edges are stored as small key-value json files.
   - The directory may contain a large number of files.
 2. Convert graph to PTBG input format.
   - PTBG requires that we index and partition all nodes and edges.
   - Look into `tools/convert_graph_for_pytorch_biggraph` for how to do this.
 3. Create a PTBG config.
   - Specify all node / edge types
   - Specify location of all input files
   - The parameters of this config must match the options used in the conversion
 4. Launch the PTBG training cluster
   - Use 10-20 machines, too many will slow this process
   - Wait at least 5 epochs, will take days
 5. Index the resulting embeddings for use in Agatha
   - Agatha needs to know where to find each embedding, given the node name
   - Use `tools/py_scripts/ptbg_index_embeddings.py` to create a lookup table
     that maps each node name to its embedding metadata.

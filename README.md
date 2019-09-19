![pymoliere_logo][pymoliere_logo]

# Overview

Moliere is a biomedical hypothesis generation system, meaning that it collects
millions of research papers and finds fruitful new research directions. For
example, we've used an earlier version of Moliere to identify [gene-treatment 
targets for HIV-Associated Dementia][bioarxiv_paper].

This repository contains our updated version, written entirely in python and
built on top of [Redis][redis] and [Dask][dask]. As a result of these updates,
Moliere is easier to install, deploy, and query, even if you don't have a super
computer! Furthermore, we are working to incorporate state-of-the-art
techniques, such as the [SciBert][scibert] text embedding method, that had not
been available when we first started this project in late 2016.

At a high level, moliere consists of two components: construction and querying.
The former takes raw textual information from PubMed, combines it with
additional data (new data sources pending!), and creates a large complex network
structure. This process is done in a distributed fashion using Dask, and is
written to a centralized Redis database. The latter component uses that database
in order to provide new information. If you would like to run your own queries
using our existing network, you can simply download our pre-constructed database
(pending) and only use the query component.

# Installation

While we're working on packaging the module, the first thing you can do is clone
our working version, and install locally:
```bash
# Clone
git clone https://github.com/JSybrandt/PyMoliere.git ./pymoliere
# Install, will pull dependencies
pip install ./pymoliere
```

## Getting necessary datafiles:

Pymoliere depends on the pytorch [SciBert][scibert] text embedding model
`scibert-scivocab-uncased` that can be [downloaded here][scibert_download].
Here's some steps to download, extract, and prepare this model.

```bash
cd <location you want to store the model>
# Download model file
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
# Extract the model
tar xvf scibert_scivocab_uncased.tar
# Decompress the weights
tar zxvf scibert_scivocab_uncased/weights.tar.gz -C scibert_scivocab_uncased
# Rename the config to an expected name for pytorch
mv scibert_scivocab_uncased/bert_config.json scibert_scivocab_uncased/config.json
```

In addition to the scibert model, you will also need to keep track of the
stopword list that came along with this module. It can be found at:
```bash
.../pymoliere/data/stopwords/stopword_list.txt
```

# Prepare System

In order to run PyMoliere, you will need to setup a [Dask][dask] cluster and a
[Redis][redis] server. Note that provided a large enough compute node, both of
these tasks may be running locally. If running remotely, make sure that all
nodes within the cluster have access to the redis server. Additionally, make
sure that all compute nodes have access to the necessary data files via the same
network file system.

Moliere depends on a significantly large set of config parameters. These are all
defined using a proto specification, and can be customized through both config
files and command-line arguments. Take a look at the global defaults in
[config.proto][config_path], and see how they can be customized in some of our
example config files in the root directory. Note that the precedence order is:
global default, then config file, then command-line argument.

# Construct a Moliere Network

To begin, we recommend creating a minimal config file to specify your custom
paths to data files. For instance, it will likely look like this:
```proto
cluster {
  shared_scratch: <path to shared network working directory>
  local_scratch: <path to machine-local directory such as "/tmp">
  address: <hostname of dask scheduler>
}
parser {
  scibert_data_dir: <path to shared location of scibert data>
  stopword_list: <path to shared location of stopword list>
}
db {
  address: <hostname of redis db>
}
```

Assuming this file were named `example.conf`, you could then running the moliere
construction process is as simple as:

```bash
python3 -m pymoliere.construct example.conf
```

# Query a Moliere Network

Assuming the network has been constructed, or you have loaded a pretrained
version of the Moliere network, running a query is simple. First, you need to
identify two items of interest and determine their key in redis. If you are
unsure, the query process will be able to suggest potential keys given a
substring. Example keys include:

```
's:31522890:1:3' - the 3'rd sentence of PMID 31522890 (version 1)
'l:noun:cancer' - the term "cancer" appearing as a noun
'e:carbonic_anhydrase_inhibitor' - the entity "carbonic anhydrase inhibitor"
'm:d003430' - the mesh term "D003430" applying to "Cross-Sentional Studies"
```

Given two keys of interest, and provided the redis db is running locally,
running a query is as simple as:

```bash
python3 -m pymoliere.query --source <key 1> --target <key 2>
```

Remote database connections can be specified by either the config file line:
```proto
db { address: <db location> }
```
or the command-line argument
```
python3 -m pymoliere.query --source <key 1> --target <key 2> --db.address <address>
```

Other interesting parameters can be found in the [config.proto][config_path]
description.

# Running on the Clemson Palmetto Cluster

Palmetto has some weird quirks. Firstly, here are the required module files:
```bash
module load gcc/8.2.0
module load cuda-toolkit/9.0.176
module load cuDNN/9.0v7
```

Setting up a dask cluster first requires that you request a set of nodes:
```bash
qsub -I -l select=100:ncpus=16:mem=60gb,walltime=72:00:00
```
Then, once you've been given the interactive job, take a note of your hostname
and run the following:
```bash
echo $HOSTNAME  # will output something like node####.palmetto.clemson.edu
dask-ssh --hostfile $PBS_NODEFILE
```

Note that *EVERY* compute machine is going to need to have the same modules and
environment loaded. You can accomplish this in your bashrc.

Once the cluster is running, you can get a node to start the computation and
host the redis server. I recommend making this node bigger than the compute
nodes in the dask cluster.
```bash
qsub -I -l select=1:ncpus=24:mem=120gb,walltime=72:00:00
# Start redis server
<install_path>/redis-server <git clone path>/scripts/moliere.redis.conf
```

Now you are free to run pymoliere!

# Cite Us

```
@inproceedings{sybrandt2017,
 author = {Sybrandt, Justin and Shtutman, Michael and Safro, Ilya},
 title = {MOLIERE: Automatic Biomedical Hypothesis Generation System},
 booktitle = {Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 series = {KDD '17},
 year = {2017},
 isbn = {978-1-4503-4887-4},
 location = {Halifax, NS, Canada},
 pages = {1633--1642},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3097983.3098057},
 doi = {10.1145/3097983.3098057},
 acmid = {3098057},
 publisher = {ACM},
 address = {New York, NY, USA},
}

@inproceedings{sybrandt2018a,
 author = {Sybrandt, Justin and Shtutman, Micheal and Safro, Ilya},
 booktitle = {2018 IEEE International Conference on Big Data (Big Data)},
 doi = {10.1109/bigdata.2018.8622637},
 pages = {1494-1503},
 title = {Large-Scale Validation of Hypothesis Generation Systems via Candidate Ranking},
 year = {2018}
}

@inproceedings{sybrandt2018b,
 author = {Sybrandt, Justin and Carrabba, Angelo and Herzog, Alexander and Safro, Ilya},
 booktitle = {2018 IEEE International Conference on Big Data (Big Data)},
 doi = {10.1109/bigdata.2018.8621974},
 pages = {1504-1513},
 title = {Are Abstracts Enough for Hypothesis Generation?},
 year = {2018}
}
```


[pymoliere_logo]:https://github.com/JSybrandt/PyMoliere/raw/master/pymoliere_header.png
[scibert]:https://github.com/allenai/scibert
[scibert_download]:https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
[dask]:https://dask.org/
[redis]:https://redis.io/
[bioarxiv_paper]:https://www.biorxiv.org/content/10.1101/591438v3
[config_path]:https://github.com/JSybrandt/PyMoliere/blob/master/pymoliere/config/config.proto

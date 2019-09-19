![pymoliere_logo][pymoliere_logo]


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

# Construct a Moliere Network

# Query a Moliere Network

# Running on the Clemson Palmetto Cluster


[pymoliere_logo]:https://github.com/JSybrandt/PyMoliere/raw/master/pymoliere_header.png
[scibert]:https://github.com/allenai/scibert
[scibert_download]:https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar

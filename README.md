![logo](https://github.com/JSybrandt/agatha/blob/master/logo.png?raw=true)

# Install Agatha to use pretrained models

In [our paper][paper_link] we present state-of-the-art performance numbers
across a range of recent biomedical discoveries across popular biomedical
sub-domains. We trained the Agatha system using only data published prior to
2015, and supply the necessary subset of that data in an easy-to-replicate
package. Note, the full release is also available for those wishing to tinker
further. Here's how to get started.

Setup a conda environment

```
conda create -n agatha python=3.8
conda activate agatha
```

Install PyTorch. We need a version >= 1.4, but different systems will require
different cuda library versions. We installed PyTorch using this command:

```
conda install pytorch cudatoolkit=9.2 -c pytorch
```

__Temporary:__ Due to incorrectly configured dependencies of pytorch-lightning
for version 0.6.0, we have to install their latest release manually for the
time-being.

```
pip install git+https://github.com/PyTorchLightning/pytorch-lightning.git
```


Install Agatha. This comes along with the dependencies necessary to run the
pretrained model. Note, we're aware of a pip warning produced by this install
method, we're working on providing a easier pip-installable wheel.

```
pip install git+https://github.com/JSybrandt/agatha.git
```

Now we can download the 2015 hypothesis prediction subset.  Note, at the time of
writing, we only provide a 2015 validation version of Agatha. We are in the
process of preparing an up-to-date 2020 version.  We recommend the tool `gdown`
that comes along with Agatha to download our 38.5GB file. If you don't want to
use that tool, you can download the same file from your browser via [this
link][2015_model_link].

```
# Remeber where you place your file
cd <AGATHA_DATA_DIR>
# This will place 2015_hypothesis_predictor_512.tar.gz in AGATHA_DATA_DIR
gdown --id 1Tka7zPF0PdG7yvGOGOXuEsAtRLLimXmP
# Unzip the download, creates hypothesis_predictor_512/...
tar -zxvf 2015_hypothesis_predictor_512.tar.gz
```

We can now load the Agatha model in python. After loading, we need to inform the
model of where it can find its helper data. By default it looks in the current
working directory.

```python3
# We need to load the pretrained agatha model.
import torch
torch.load("<AGATHA_DATA_DIR>/hypothesis_predictor_512/model.pt")

# We need to tell the model abouts its helper data.
model.set_data_root("<AGATHA_DATA_DIR>/hypothesis_predictor_512/")

# We need to setup the internal datastructures around that helper data.
model.init()

# Now we can run queries specifying two umls terms! Note, this process has some
# random smapling involved, so your result might not look exactly like what we
# show here.
# Kewords:
#   Cancer: C0006826
#   Tobacco: C0040329
model.predict_from_terms([("C0006826", "C0040329")])
>>> [0.78358984]

# Kewords:
#   Cancer: C0006826
#   Tobacco: C0040329
model.predict_from_terms([("C0006826", "C0040329")])
>>> [0.78358984]

# If you want to run loads of queries, we recommend first using
# model.init_preload(), and then the following syntax. Note that
# predict_from_terms will automatically compute in batches of size:
# model.hparams.batch_size.
queries = [("C###", "C###"), ("C###", "C###"), ..., ("C###", "C###")]
model = model.eval()
model = model.cuda()
with torch.no_grad():
  predictions = model.predict_from_terms(queries)
```

# Replicate the 2015 Validation Experiments

Provided in `./benchmarks` are the files we use to produce the results found in
our paper. Using the 2015 pretrained model, you should be able to replicate
these results. This guide focuses on the recommendation experiments, wherein all
pairs of elements from among the 100 most popular new predicates per-subdomain
are evaluated by the Agatha model. For each of the 20 considered types, we
generated all pairs, and removed any pair that is trivially discoverable from
within the Agatha semantic graph. The result are a list of predicates in the
following json file: `./benchmarks/all_pairs_top_20_types.json`

The json predicate file has the following schema:
```
{
  "<type1>:<type2>": [
    {
      "source": "<source keyword>",
      "target": "<target keyword>",
      "label": 0 or 1
    },
    ...
  ]
  ...
}
```

Here's how to load the pretrained model and evaluate the provided set of
predicates:

```python
import torch
import json

# Load the pretrained model
model = torch.load("<AGATHA_DATA_DIR>/hypothesis_predictor_512/model.pt")
# Configure the helper data
model.set_data_root("<AGATHA_DATA_DIR>/hypothesis_predictor_512")
# Initialize the model for batch processing
model.init_preload()


# Load the json file
with open("<AGATHA_INSTALL_DIR>/benchmarks/all_pairs_top_20_types") as file:
  types2predicates = json.load(file)

# prepare model
model = model.eval()
model = model.cuda()
with torch.no_grad():

  # Predict ranking criteria for each predicate
  types2predictions = {}
  for typ, predicates in types2predictions.items():
    types2predictions[typ] = model.predict_from_terms([
      (pred["source"], pred["target"])
      for pred in predicates
    ])
```


# Installing Agatha for Development

These instructions are useful if you want to customize Agatha, especially if you
are also running this system on the Clemson Palmetto Cluster. This guide
also assumes that you have already installed `anaconda3`.

Step zero. Get yourself a node made in the last few years with a decent GPU.
Currently supported GPU's on palmetto include the `P100` and the `V100`. Recent
changes to pytorch are incompatible with older models.

The recommended node request is:
```
qsub -I -l select=5:ncpus=40:mem=365gb:ngpus=2:gpu_model=v100,walltime=72:00:00
```

First, load the following modules:
```
module load gcc/8.3.0          \
            cuDNN/9.2v7.2.1    \
            sqlite/3.21.0      \
            cuda-toolkit/9.2   \
            nccl/2.4.2-1       \
            hdf5/1.10.5        \
            mpc/0.8.1
```

Second, setup an anaconda environment for Agatha:
```
conda create -n agatha python=3.8
conda activate agatha
```

Third, install pytorch. Note, because of limitations on palmetto, you're going
to need to stick with the cudatoolkit 9.2 version.
```
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

Fourth, we can clone and install the _full_ version of Agatha.
```
# Downloads agatha source
git clone https://github.com/JSybrandt/agatha.git
cd agatha
# Installs the agatha module
pip install -e .
# Installs the developer requirements
pip install -r requirements.txt
```

Now you should be ready to roll! I recommend you create the following file
somewhere like `~/prep_agatha_env`:

```
# Remove current modules (if any)
module purge
# Leave current conda env (if any)
conda deactivate
# Load all nessesary palmetto modules
module load gcc/8.3.0 mpc/0.8.1 cuda-toolkit/9.2 cuDNN/9.2v7.2.1 nccl/2.4.2-1 \
            sqlite/3.21.0 hdf5/1.10.5
# Include hdf5, needed to build tools
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/software/hdf5/1.10.5/include
# Load python modules
conda activate agatha
```

You can then switch to your agatha environment, with modules loaded, by running:
```
source ~/prep_agatha_env
```


[paper_link]:https://arxiv.org/abs/2002.05635
[2015_model_link]:https://drive.google.com/uc?id=1Tka7zPF0PdG7yvGOGOXuEsAtRLLimXmP


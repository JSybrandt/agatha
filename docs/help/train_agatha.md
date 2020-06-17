How to Train Agatha
===================

Training the Agatha deep learning model is the last step to generating
hypotheses after you've already processed all necessary information using
`agatha.construct`. This process uses [PyTorch][1] and [PyTorch-Lightning][2] to
efficiently manage the distributed training of the predicate ranking model
stored in `agatha.ml.hypothesis_predictor`.

## tl:dr;

You will need the following files:

 - `predicate_graph.sqlite3`
 - `predicate_entities.sqlite3`
 - `embeddings/predicate_subset/*.h5`

You will need to run `python3 -m agatha.ml.hypothesis_predictor` with the right
keyword arguments. If performing distributed training, you will need to run this
on each machine in your training cluster.

Take a look at
[scripts/train_2020.sh][https://github.com/JSybrandt/agatha/blob/master/scripts/train_2020.sh]
for how to train the agatha model.

If you are running the training process on one machine and only one gpu, you
will want to remove the `distributed_backend` flag, and make sure `num_nodes` is
set to one. If you are using multiple machines, or _multiple gpus on one
machine_, then you will want to make sure that `distributed_backend="ddp"` and
you should take a look at setting the distributed environment variables if you
run into errors. In the multi-gpu one-machine case, these variables should be
set automatically.


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
documentation for more detail.

## Installing Apex for AMP

Apex is a bit of a weird dependency, but it allows us to take advantage of some
GPU optimizations that really cut back our memory footprint. Amp allows us to
train using 16-bit precision, enabling more samples per batch, resulting in
faster training times. However, note that if you install apex on a node that has
one type of GPU, you will get an error if you try and train on another. This
means that you **need** to install this dependency on a training node with the
appropriate GPU.

Warning: Apex is going to require a different version of GCC than we typically
use. If you're on palmetto, you can run: `module rm gcc/8.1.0; module load
gcc/6.3.0`

To install apex, first select a location such as `~/software` to keep the files.
Next, download apex with git `git clone https://github.com/NVIDIA/apex.git`.
Finally, install the dependency with: `pip install -v --no-cache-dir
--global-option="--cpp_ext" --global-option="--cuda_ext" ./`

In full, run this:

```bash
# SSH into one of your training nodes with the correct GPU configuration.
# Make sure the appropriate modules are loaded.
# Assuming you would like to install apex in ~/software/apex

# make software dir if its not present
mkdir -p ~/software  

# Clone apex to ~/software/apex
git clone https://github.com/NVIDIA/apex.git ~/software/apex

# Enter Apex dir
cd ~/software/apex

# Run install
pip install -v                      \
  --no-cache-dir                    \
  --global-option="--cpp_ext"       \
  --global-option="--cuda_ext"      \
  ./
```

## Model Parameters

This is _NOT_ an exhaustive list of the parameters present in the Agatha deep
learning model, but is a full list of the parameters you need to know to train
the model.

`amp_level` 
: The optimization level used by NVIDIA. `O1` works well. `O2` causes some
  convergence issues, so I would stay away from that.

`default_root_dir` 
: The directory to store model training files.

`dataloader-workers` 
: The number of processes used to generate predicate pairs, per-gpu. Too many
  dataloader workers will cause an out-of-memory error. I've found 3 works well.

`dim` 
: The number of dimensions of each input embedding. We use 512 in most cases.
  This parameter effects the size of various internal parameters.

`distributed_backend` 
: Used to specify how to communicate between GPUs. Ignored if using only one
  GPU. Set to `ddp` for distributed data parallel (even if only using gpus on
  the same node).

`embedding-dir` 
: The system path containing embedding `HDF5` (`*.h5`) files.

`entity-db` 
: The system path to the entities `.sqlite3` database.

`gpus` 
: The specific GPUs enabled on this machine. GPUs are indexed starting from 0.
  On a 2-GPU node, this should be set to `0,1`.

`gradient_clip_val` 
: A single step of gradient decent cannot move a parameter more than this
  amount. We find that setting this to `1.0` enables convergence.

`graph-db` 
: The system path to the graph `.sqlite3` database.

`lr` 
: The learning rate. We use `0.02` because we're cool.

`margin` 
: The objective of the Agatha training procedure is [Margin Ranking Loss][3].
  This parameter determines how different a positive ranking criteria needs to
  be from all negative ranking criteria. Setting this too high or low will cause
  convergence issues. Remember that the model outputs in the `[0,1]` interval.
  We recommend `0.1`.

`max_epochs` 
: The maximum number of times to go through the training set.

`negative-scramble-rate` 
: For each positive sample, how many negative scrambles (easy negative samples).

`negative-swap-rate` 
: For each positive sample, how many negative swaps (hard negative samples).

`neighbor-sample-rate` 
: When sampling a term-pair, we also sample each pair's disjoint neighborhood.
  This determines the maximum number of neighbors to include.

`num_nodes` 
: This determines the number of _MACHINES_ used to train the model.

`num_sanity_val_steps` 
: Before starting training in earnest, we can optionally take a few validation
  steps just to make sure everything has been configured properly. If this is
  set above zero, we will run multiple validation steps on the newly
  instantiated model. Recommended to run around `3` just to make sure everything
  is working.

`positives-per-batch` 
: Number of positive samples per batch per machine. More results in faster
  training. Keep in mind that the true batch size will be `num_nodes *
  positives-per-batch * (negative-scramble-rate + negative-swap-rate)`. When
  running with 16-bit precision on V100 gpus, we can handle around `80`
  positives per batch.

`precision` 
: The number of bits per-float. Set to `16` for half-precision if you've
  installed apex.

`train_percent_check` 
: Limits the number of actual training examples per-epoch. If set to `0.1` then
  one epoch will occur after every 10\% of the training data. This is important
  because we only checkpoint after every epoch, and don't want to spend too 
  much time computing between checkpoints. We recommend that if you set this 
  value, you should increase `max_epochs` accordingly.
 
`transformer-dropout` 
: Within the transformer encoder of Agatha, there is a dropout parameter that
  helps improve performance. Recommended you set this to `0.1`.

`transformer-ff-dim` 
: The size fo the transformer-encoded feed-forward layer. Recommended you set
  this to something between `2*dim` and `4*dim`.

`transformer-heads` 
: The number of self-attention operations per self-attention block in the
  transformer encoder. We use `16`.

`transformer-layers` 
: The number of transformer encoder blocks. Each transformer-encoder contains
  multi-headed self-attention and a feed-forward layer. More transformer encoder
  layers should lead to higher quality, but will require additional training
  time and memory.

`val_percent_check` 
: Just like how `train_percent_check` limits the number of training samples
  per-epoch, `val_percent_check` limits the number of validation samples
  per-epoch. Recommended that if you set one, you set the other accordingly.

`validation-fraction` 
: Before training, this parameter determines the training-validation split. A
  higher value means less training data, but more consistent validation numbers.
  Recommended you set to `0.2`.

`warmup-steps` 
: Agatha uses a gradient warmup strategy to improve early convergence. This
  parameter indicates the number of steps needed to reach the input learning
  rate. For instance, if you specify a learning rate of `0.02` and `100` warmup
  steps, at step `50` there will be an effective learning rate around `0.01`. We
  set this to `100`, but higher can be better if you have the time.

`weight-decay` 
: Each step, the weights of the agatha model will be moved towards zero at this
  rate. This helps with latter convergence and encourages sparsity. We set to
  `0.01`.

`weights_save_path` 
: The result root directory. Model checkpoints will be stored in
  `weights_save_path/checkpoints/version_X/`. Recommended that this is set to
  the same value as `default_root_dir`.

## Subset Data with Percent Check Flags

In the list of model flags are two that deserve more explanation:
`train_percent_check`, and `val_percent_check`. When debugging the model
training process to ensure everything has been setup correctly, it is worthwhile
to run the training routine through a couple of epochs quickly. This will ensure
that the model output checkpoints are created properly. To do so, set
`train_percent_check` and `val_percent_check` to a very small value, such as
`0.0001`. Preferably, this will be small enough to complete an epoch in a couple
of minutes. Warning, you set this value too low, you will filter out _all_ of
the training data and will create problems.

When you _actually_ want to train the model, you still might want a modest
`train_percent_check` and `val_percent_check`. For instance, if the estimated
time per epoch is greater than a couple of hours, you might want more frequent
check pointing. What we want to avoid is the amount of training time that is
lost when an unpredictable system failure causes an outage 40 hours into
training, and we haven't created our first checkpoint yet. If this were to
happen, we would simply lose all of the progress we had made for nearly two days
worth of computational effort.

Therefore, I recommend setting these values to something that reduces the time
per epoch to the single-digit hours. Keep in mind that when you reduce the
training set, and especially when you reduce the validation set, you should
expect poorer convergence in the final model. Therefore, if at all possible, it
is recommend that you increase the number of training processes by adding more
distributed workers. Once you have as many machines as you can afford, then tune
this parameter.

## Running Distributed Training

In order to preform distributed training, you will need to ensure that your
training cluster is each configured with the same modules, libraries, and python
versions.

On palmetto, and many HPC systems, this can be done with modules and Anaconda. I
recommend adding a section to your `.bashrc` for the sake of training Agatha
that loads all necessary modules and activates the appropriate conda
environment. As part of this configuration, you will need to set some
environment variables on each machine that help coordinate training. These are `MASER_ADDR`, `MASTER_PORT`, and `NODE_RANK`. 

### Distributed Training Env Variables

`MASER_ADDR`
: Needs to be set to the hostname of one of your training nodes. This node will
  coordinate the others.

`MASTER_PORT`
: Needs to be set to an unused network port for each machine. Can be any large
number. We recommend: `12910`.

`NODE_RANK`
: If you have N machines, then each machine needs a unique `NODE_RANK` value
  between 0 and N-1.

We recommend setting these values automatically using a `nodefile`. A `nodefile`
is just a text file containing the hostnames of each machine in your training
cluster. The first name will be the `MASTER_ADDR` and the `NODE_RANK` will
correspond to the order of names in the file.

If `~/.nodefile` is the path to your nodefile, then you can set these values
with:

```bash
export NODEFILE=$HOME/.nodefile
export NODE_RANK=$(grep -n $HOSTNAME $NODEFILE | awk 'BEGIN{FS=":"}{print $1-1}')
export MASTER_ADDR=$(head -1 $NODEFILE)
export MASTER_PORT=12910
```

If you're on palmetto, you've already got access to the nodefile referenced by
`PBS_NODEFILE`. However, only the first machine will have this variable set. I
recommend automatically copying this file to some shared location whenever it is
detected. You can do that in `.bashrc` by putting the following lines _BEFORE_
setting the `NODE_RANK` and  `MASER_ADDR` variables.

```bash
# If $PBS_NODEFILE is a file
if [[ -f $PBS_NODEFILE ]]; then
  cp $PBS_NODEFILE ~/.nodefile
fi
```

### Launching Training on Each Machine with Parallel

Once each machine is configured, you will then need to run the agatha training
module on each. We recommend `parallel` to help you do this. Parallel runs a
given bash script multiple times simultaneously, and has some flags that let
us run a script on each machine in a nodefile.

Put simply, you can start distributed training with the following:

```bash
parallel \
  --sshloginfile $NODEFILE \
  --ungroup \
  --nonall \
  python3 -m agatha.ml.hypothesis_predictor \
    ... agatha args ...
```

To explain the parameters:

`sshloginfile`
: Specifies the set of machines to run training on. We use the `NODEFILE` 
  created in the previous step.

`ungroup`
: By default, `parallel` will wait until a process exits to show us its output.
  This flag gives us input every time a process writes the newline character.

`nonall`
: This specifies that the following command (`python3`) will not need its
  arguments set by `parallel`, and that we would like to run the following
  command as-is, once per machine in `$NODEFILE`.

## Palmetto-Specific Details

On palmetto, there are a number of modules that you will need to run Agatha.
Here is what I load on every machine I use to train agatha:

```bash
# C++ compiler modules
module load gcc/8.3.0
module load mpc/0.8.1

# NVIDIA modules
module load cuda-toolkit/10.2.89
module load duDNN/10.2.v7.6.5
module load nccl/2.6.4-1

# Needed for parallel
module load gnu-parallel

# Needed to work with HDF5 files
module load hdf5/1.10.5

# Needed to work with sqlite
module load sqlite/3.21.0

conda activate agatha

# Copy PBS_NODEFILE if it exists
if [[ -f $PBS_NODEFILE ]]; then
  cp $PBS_NODEFILE ~/.nodefile
fi

# Set distributed training variables
export NODEFILE="~/.nodefile"
export NODE_RANK=$(grep -n $HOSTNAME $NODEFILE | awk 'BEGIN{FS=":"}{print $1-1}')
export MASTER_ADDR=$(head -1 $NODEFILE)
export MASTER_PORT=12910
```

# Loading the Trained Model

Once you've completed a few epochs of training, you will hopefully a see a file
appear in
`{weights_save_path}/lightning_logs/version_{#}/checkpoints/epoch={#}.ckpt`

If course, `weights_save_path` refers to whatever directory you listed in
`--weights_save_path` in the training command-line arguments. The version number
refers to the model version that pytorch-lightning deduces while training. Each
time you run the training script with the same checkpoint directory, this number
will increment. Then the epoch number will refer to whatever epoch this model
last updated its checkpoint. Note here that the epoch number might be less than
the number of epochs you've actually computed, because we will only update the
checkpoint when the validation loss is improved.

To load the checkpoint in python, use:

```python3
from agatha.ml.hypothesis_predictor import HypothesisPredictor
model = HypothesisPredictor.load_from_checkpoint( ... )
```

When you want to give this model to someone else, you often don't want to give
them the whole checkpoint. For this, you can use a simpler pytorch model format.
The conversion is really simple:

```python3
checkpoint_path = ...
output_path = ...
import torch
from agatha.ml.hypothesis_predictor import HypothesisPredictor

# Load model from checkpoint
model = HypothesisPredictor.load_from_checkpoint(checkpoint_path)
# Save model in pytorch model format
torch.save(model, output_path)
```

The reason to do this is so future users can load your model with:

```python3
import torch
model = torch.load(...)
```

## Running your new model.

Now that you have a model that you can load (either through
`load_from_checkpoint` or `torch.load`, you can run some examples to ensure that
everything has been configured properly. The simplest way to do this is to run a
little script like this in your python terminal:

```python3
from agatha.ml.hypothesis_predictor import HypothesisPredictor
model = HypothesisPredictor.load_from_checkpoint("...")
# - OR -
import torch
model = torch.load("...")

# Configure auxilary data paths
model.configure_paths(
  embedding_dir="/path/to/embeddings",
  entity_db="/path/to/entities.sqlite3",
  graph_db="/path/to/graph.sqlite3",
)

# Optional, if you're going to do a lot of queries.
model = model.eval()
model.preload()

# C0006826 is the term for Tobacco
# C0040329 is the term for Cancer
print(model.predict_from_terms([("C0006826", "C0040329")]))
```

If this outputs something like `[0.9]` (or any other float, if your model hasn't
really been trained), then you're good!

[1]:https://pytorch.org/
[2]:https://github.com/PytorchLightning/pytorch-lightning
[3]:https://pytorch.org/docs/stable/nn.html#torch.nn.MarginRankingLoss
[4]:https://www.sqlite.org/index.html

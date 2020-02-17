![logo](https://github.com/JSybrandt/agatha/blob/master/logo.png?raw=true)

[Docs remain a work in progress]

# Installing Agatha to Run Pretrained Models

Note: Models have not yet been posted.

In order to run our experiments yourself, you can pip install this repo:

```
pip install git+https://github.com/JSybrandt/agatha.git
```

All python modules to load the supplied checkpoints and run your own experiments
will be loaded.

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

#!/bin/bash
#PBS -l select=50:ncpus=24:mem=125gb:ngpus=2:gpu_model=k40,walltime=72:00:00
#PBS -o /home/jsybran/logs/train_ab_gen.out
#PBS -e /dev/null
#PBS -N train_ab_gen

CONFIG=$1

rm -rf /scratch4/jsybran/.torch_dist_init

if [[ ! -f $CONFIG ]]; then
  echo "Must supply config as \$1"
  exit 1
fi

NODEFILE=/home/jsybran/.nodefile
if [[ ! -f "$NODEFILE" ]]; then
  echo "Must supply a nodefile at ~/.nodefile"
  exit 1
fi

NUM_NODES=$(wc -l < $NODEFILE)
echo Starting on $NUM_NODES nodes

MAX_RANK=$(($NUM_NODES-1))

parallel \
  --sshloginfile "$NODEFILE" \
  --ungroup \
  -j 1 \
  """
    echo Starting {} on \$HOSTNAME
    torchbiggraph_train --rank {} $CONFIG
  """ ::: $(seq 0 $MAX_RANK)

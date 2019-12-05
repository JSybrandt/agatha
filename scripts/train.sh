#!/bin/bash
#PBS -l select=50:ncpus=24:mem=125gb:ngpus=2:gpu_model=k40,walltime=72:00:00
#PBS -o /home/jsybran/logs/train_ab_gen.out
#PBS -e /dev/null
#PBS -N train_ab_gen

NODEFILE=$PBS_NODEFILE
if [[ ! -f "$NODEFILE" ]]; then
  NODEFILE=/home/jsybran/.nodefile
fi
if [[ ! -f "$NODEFILE" ]]; then
  echo "Must supply a nodefile"
  exit 1
fi

NUM_NODES=$(wc -l < $NODEFILE)
echo Starting on $NUM_NODES nodes

parallel \
  --nonall \
  --sshloginfile "$NODEFILE" \
  --ungroup \
  python3 -m pymoliere.ml.abstract_generator /zfs/safrolab/users/jsybran/pymoliere/configs/abstract_generator.conf --num_nodes $NUM_NODES

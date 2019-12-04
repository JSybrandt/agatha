#!/bin/bash
#PBS -l select=50:ncpus=24:mem=125gb:ngpus=2:gpu_model=k40,walltime=72:00:00
#PBS -o /home/jsybran/logs/train_ab_gen.out
#PBS -e /dev/null
#PBS -N train_ab_gen

parallel \
  --nonall \
  --linebuffer \
  --sshloginfile "$PBS_NODEFILE" \
  -j 1 \
  python3 -m pymoliere.ml.abstract_generator /zfs/safrolab/users/jsybran/pymoliere/configs/abstract_generator.conf

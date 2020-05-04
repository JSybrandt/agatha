#!/bin/bash

# This script trains the Agatha deep-learning model on the Clemson Palmetto
# Cluster. It is included so that you can see an example of how to set the
# training parameters.

DISTRIBUTED=${1:-"0"}

#RELEASE_DIR="/zfs/safrolab/users/jsybran/agatha/data/releases/2020"
RELEASE_DIR="/burstbuffer/fast/covid/2020_release"
MODEL_DIR="$RELEASE_DIR/hypothesis_predictor"
mkdir -p $MODEL_DIR

NUM_NODES=1
GPUS="0,1"
DIST_BACKEND="ddp"
VERBOSE_FLAG="--verbose"
FRAC_PER_EPOCH=0.1
VAL_FRAC=0.2

# DISTRIBUTED
if [[ $DISTRIBUTED != "0" ]]; then
  NODEFILE=/home/jsybran/.nodefile
  NUM_NODES=$(wc -l < $NODEFILE)
fi


CMD="""
python3 -m agatha.ml.hypothesis_predictor                                     \
  --amp_level O1                                                              \
  --default_root_dir $MODEL_DIR                                               \
  --dataloader-workers 3                                                      \
  --dim 512                                                                   \
  --distributed_backend $DIST_BACKEND                                         \
  --embedding-dir $RELEASE_DIR/embeddings/predicate_subset                    \
  --entity-db $RELEASE_DIR/predicate_entities.sqlite3                         \
  --gpus $GPUS                                                                \
  --gradient_clip_val 1.0                                                     \
  --graph-db $RELEASE_DIR/predicate_graph.sqlite3                             \
  --lr 0.02                                                                   \
  --margin 0.1                                                                \
  --max_epochs 10                                                             \
  --negative-scramble-rate 10                                                 \
  --negative-swap-rate 30                                                     \
  --neighbor-sample-rate 15                                                   \
  --num_nodes $NUM_NODES                                                      \
  --num_sanity_val_steps 3                                                    \
  --positives-per-batch 80                                                    \
  --precision 16                                                              \
  --train_percent_check $FRAC_PER_EPOCH                                       \
  --transformer-dropout 0.1                                                   \
  --transformer-ff-dim 1024                                                   \
  --transformer-heads  16                                                     \
  --transformer-layers 4                                                      \
  --val_percent_check $FRAC_PER_EPOCH                                         \
  --validation-fraction $VAL_FRAC                                             \
  --warmup-steps 100                                                          \
  --weight-decay 0.01                                                         \
  --weights_save_path $MODEL_DIR                                              \
  $VERBOSE_FLAG
"""

if [[ $DISTRIBUTED != "0" ]]; then
  echo "Launching Distributed Training across $NUM_NODES machines"
  parallel \
    --sshloginfile "$NODEFILE" \
    --ungroup \
    --nonall \
    $CMD
else
  $CMD
fi

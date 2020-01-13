#!/bin/bash

export OMP_NUM_THREADS=24

CONFIG=$1

RANK=$(nl ~/.nodefile | grep $HOSTNAME | awk '{print $1-1}')
echo Starting $RANK on $HOSTNAME
torchbiggraph_train --rank $RANK $CONFIG


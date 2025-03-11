#!/usr/bin/env bash

export CUBLAS_WORKSPACE_CONFIG=:16:8
eval "$(conda shell.bash hook)"
conda activate pt


echo $PWD
# echo `nvcc --version`

model=${1:-roberta}
path=$2{:-./_runs}

echo "***** Running on node: `hostname -a` *****"
cluster=`hostname -a`
echo "***** Model: $model *****" 
echo ""

python train.py --model $model --path $path
#!/bin/bash

output_dir="${1}"
shift 1; args=$(echo "$*")
echo "Args passed to bash ${0##*/}:"
echo "=> ${args}"

# initialize the conda environment
source ./scripts/setup_env.sh

umask 002
mkdir -p ${output_dir}

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=${N_GPUS} main_dune.py  \
    --output_dir=${output_dir} \
    --seed=${RANDOM} \
    ${args}
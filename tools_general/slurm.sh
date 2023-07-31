#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
NAME=$3
CONFIG=$4
WORK_DIR=$5
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-"-t 7-0"}
TYPE=${TYPE:-"v100"}
PORT=${PORT:-$((28500 + $RANDOM % 2000))}
PY_ARGS=${@:6}

MASTER_PORT=${PORT} PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${TYPE}:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/${NAME}.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" --seed 0 --deterministic ${PY_ARGS}

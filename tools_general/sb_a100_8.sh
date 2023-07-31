#!/bin/bash
#------------------------------------------------------#
#              Edit Job specifications                 #    
#------------------------------------------------------#
#SBATCH --job-name=hb_a100_8
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --time=0-23:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --reservation=A100


set -x

NAME=$1
CONFIG=$2
WORK_DIR=$3
PORT=${PORT:-28500}
PY_ARGS=${@:4}

MASTER_PORT=${PORT} PYTHONPATH="./":$PYTHONPATH \
    srun --output=${WORK_DIR}/slurm_${NAME}.out \
    python -u tools/${NAME}.py \
    ${CONFIG} --work-dir=${WORK_DIR} \
    --launcher="slurm" \
    --seed 0 --deterministic \
    ${PY_ARGS}

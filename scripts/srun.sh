#!/bin/bash
set -x

export MASTER_PORT=29502
export NUM_GPUS=8

srun --partition llm4 --gres=gpu:$NUM_GPUS --quotatype=reserved --job-name vlmeval --cpus-per-task=24 \
    torchrun --nproc-per-node=$NUM_GPUS --master_port ${MASTER_PORT} run.py \
    --verbose \
    --data AI2D_TEST \
    --model Yi_34B \
    --max-new-tokens 128
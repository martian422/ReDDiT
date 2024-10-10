#!/bin/bash
set -x
WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-23456}

pkill -f torchrun
pkill -f main.py
pkill -f wandb
sleep 10s
nvidia-smi
NGPUS=$(python -c "import torch; print(torch.cuda.device_count())")

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=INFO
export OMP_NUM_THREADS=4

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/project/zxs/datasets/huggingface
export TORCH_HOME=/share/project/zxs/.cache/torch

export WANDB_KEY=""
export ENTITY=""
export PROJECT="mdlm"
RANDOM_STRING=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 8)

ulimit -n 65536

torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=$NGPUS \
    --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT main.py \
    model=1B \
    data=gsm8k \
    data.cache_dir=/share/project/zxs/project/mdlm/data \
    wandb.name=mdlm-1B-gsm8k-${RANDOM_STRING} \
    parameterization=subs \
    model.length=512 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=$WORLD_SIZE \
    loader.num_workers=16 \
    loader.batch_size=2 \
    loader.global_batch_size=256

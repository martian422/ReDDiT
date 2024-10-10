#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export WANDB_DISABLED=true

export HF_ENDPOINT=https://hf-mirror.com

export PYTHONPATH=$PYTHONPATH:/home/MaTianren/Workspace/MDLM-neo

RANDOM_STRING=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 8)

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=1 \
    --master_port=23333 \
    main.py \
    model=L-model \
    data=llamaGen \
    parameterization=subs \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=1 \
    loader.num_workers=32 \
    loader.batch_size=8 \
    loader.global_batch_size=256

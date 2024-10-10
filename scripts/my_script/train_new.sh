#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export HF_ENDPOINT=https://hf-mirror.com

export WANDB_MODE=offline

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/mdm/bin/torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=1 \
    --master_port=11451 \
    main.py \
    model=1B \
    data=llamaGen \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/datasets/cache \
    wandb.name=mdlm-1B-llamaGen-600k-128 \
    parameterization=subs \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=1 \
    loader.num_workers=32 \
    loader.batch_size=16 \
    loader.global_batch_size=256

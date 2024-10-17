#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT=https://hf-mirror.com

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=4 \
    --master_port=11454 \
    main.py \
    model=1B \
    data=llamaGen \
    data.dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/dataset_5M  \
    data.image_token_dir=/workspace/intern/liaomingxiang/ARG-MDM/image_tokens \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/data/cache \
    wandb.name=mdm-1B-5M-cosine-m1024 \
    optim.lr=3e-4 \
    data.size=5M \
    parameterization=subs \
    mask_vocab_size=1024 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=1 \
    loader.num_workers=32 \
    loader.batch_size=16 \
    loader.global_batch_size=256 \
    checkpointing.resume_ckpt_path=/workspace/intern/liaomingxiang/ARG-MDM/MDM-1010/outputs/mdm-1B-5M-cosine-m1024/2024.10.15/173320/checkpoints/2-50000.ckpt

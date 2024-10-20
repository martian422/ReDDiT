#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0


TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT=https://hf-mirror.com

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=4 \
    --master_port=11450 \
    main.py \
    model=1B \
    data=llamaGen \
    data.dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/dataset_v0  \
    data.image_token_dir=/workspace/intern/liaomingxiang/ARG-MDM/coco30k/t2i_ds16_256_s1 \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/data/cache \
    wandb.name=mdm-1B-llamaGen-30k-m128-const \
    lr_scheduler=constant_warmup \
    data.size=30k \
    parameterization=subs \
    mask_vocab_size=128 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=1 \
    loader.num_workers=32 \
    loader.batch_size=16 \
    loader.global_batch_size=256

#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT=https://hf-mirror.com

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=8 \
    --master_port=11458 \
    main.py \
    model=1B \
    data=llamaGen \
    data.dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/generated-7000-12M  \
    data.val_dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/generated-7000-12M  \
    data.image_token_dir=/workspace/intern/liaomingxiang/ARG-MDM/generated_laion_coco \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/data/cache \
    wandb.name=gen-1B-12M-m1024-s100-re \
    lr_scheduler=constant_warmup \
    optim.lr=3e-4 \
    data.size=12M \
    generation_cfg=7.5 \
    ar_cfg=False \
    parameterization=subs \
    mask_vocab_size=1024 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=100 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=16 \
    loader.global_batch_size=512

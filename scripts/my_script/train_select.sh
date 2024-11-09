#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT=https://hf-mirror.com

ulimit -n 65536

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=4 \
    --master_port=11456 \
    main.py \
    model=1B \
    data=llamaGen \
    data.dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/COCO-30000-filter55-16M  \
    data.val_dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/COCO-30000-filter55-16M  \
    data.image_token_dir=/workspace/intern/liaomingxiang/ARG-MDM/laion-coco \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/data/cache \
    wandb.name=select-1B-16M-bs768 \
    lr_scheduler=constant_warmup \
    optim.lr=5e-4 \
    data.size=16M \
    generation_cfg=3.0 \
    ar_cfg=False \
    parameterization=subs \
    mask_vocab_size=1024 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=100 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=48 \
    loader.global_batch_size=768

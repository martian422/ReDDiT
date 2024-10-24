#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export WANDB_DISABLED=true

TOKENIZERS_PARALLELISM=false

export HF_ENDPOINT=https://hf-mirror.com

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=8 \
    --master_port=11455 \
    main.py \
    model=1B \
    data=llamaGen \
    data.dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/laion-coco-2800-5M  \
    data.val_dataset_path=/workspace/intern/liaomingxiang/ARG-MDM/data/laion-coco-2800-5M  \
    data.image_token_dir=/workspace/intern/liaomingxiang/ARG-MDM/laion-coco \
    data.cache_dir=/workspace/intern/liaomingxiang/ARG-MDM/data/cache \
    wandb.name=mdm-1B-coco-5M-m1024-s100-bs512-test \
    lr_scheduler=constant_warmup \
    optim.lr=3e-4 \
    data.size=5M \
    parameterization=subs \
    mask_vocab_size=1024 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=24 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=16 \
    loader.global_batch_size=512

#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

WORLD_SIZE=1
RANK=0

TOKENIZERS_PARALLELISM=false

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=8 \
    --master_port=11456 \
    main.py \
    model=L-dit-model \
    mode=train \
    data=llamaGen-both \
    data.dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256  \
    data.val_dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256  \
    data.image_token_dir=/nfs/mtr/datasets/imagenet_10crop_code \
    data.image_dir=/nfs/mtr/datasets/imagenet_10crops_x256 \
    data.cache_dir=/data/dataset_collects/cache \
    wandb.name=mask-ddit-std-L-repa8 \
    lr_scheduler=constant_warmup \
    noise=loglinear \
    carry_over=True \
    repa_loss.use_repa=True \
    optim.lr=4e-4 \
    data.size=4M \
    time_conditioning=True \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=512 \

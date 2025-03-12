#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

WORLD_SIZE=1
RANK=0


ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=8 \
    --master_port=11456 \
    main.py \
    model=maskgit \
    mode=train \
    lm_vocab_size=16384 \
    data=llamaGen-both \
    data.dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256-sdf8-1024  \
    data.val_dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256-sdf8-1024  \
    data.image_token_dir=/nfs/mtr/datasets/imagenet_10crop_sdf8_code \
    data.image_dir=/nfs/mtr/datasets/imagenet_10crops_x256 \
    data.cache_dir=/data/dataset_collects/cache \
    wandb.name=ddit-sdf8-m1-bs1024-norepa \
    lr_scheduler=cosine_decay_warmup\
    noise=loglinear \
    carry_over=True \
    rope=2d \
    repa_loss.use_repa=False \
    repa_loss.latent_size=32 \
    optim.lr=6e-4 \
    time_conditioning=True \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=1024 \
    eval.compute_generative_perplexity=False \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=32 \
    loader.global_batch_size=1024 \

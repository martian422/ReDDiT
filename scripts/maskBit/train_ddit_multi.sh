#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MASTER_ADDR="10.10.40.232"
MASTER_PORT=11456
NNODES=2
NPROC_PER_NODE=8
NODE_RANK=$1


ulimit -n 65536

export TORCH_DISTRIBUTED_DEBUG=DETAIL

export GLOO_SOCKET_IFNAME=bond0.45
export NCCL_SOCKET_IFNAME=bond0.45
export NCCL_DEBUG=INFO  # Enable debugging logs


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    model=ddit-L \
    mode=train \
    lm_vocab_size=16384 \
    data=llamaGen-both \
    data.dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256-MaskBit  \
    data.val_dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256-MaskBit  \
    data.image_token_dir=/nfs/mtr/datasets/imagenet_10crop_maskb_code \
    data.image_dir=/nfs/mtr/datasets/imagenet_10crops_x256 \
    data.cache_dir=/data/dataset_collects/cache \
    wandb.name=L-MaskBit-frozenm128-bs1024-repa \
    lr_scheduler=cosine_decay_warmup_old \
    noise=loglinear \
    carry_over=True \
    rope=2d \
    repa_loss.use_repa=True \
    optim.lr=6e-4 \
    time_conditioning=True \
    parameterization=subs \
    mask_vocab_size=128 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    trainer.num_nodes=2 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=1024 \

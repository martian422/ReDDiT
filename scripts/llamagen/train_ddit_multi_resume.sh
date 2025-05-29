#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MASTER_ADDR="10.10.40.232"
MASTER_PORT=11456
NNODES=2
NPROC_PER_NODE=8
NODE_RANK=$1


ulimit -n 65536

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
    model=maskgit \
    mode=train \
    resume_modified_scheduler=True \
    lm_vocab_size=16384 \
    data=IBQ-both-10crop \
    wandb.name=old-ddit-maskgit-m128-10crop-conresume5e5 \
    lr_scheduler=constant_warmup \
    noise=loglinear \
    carry_over=True \
    rope=2d \
    repa_loss.use_repa=True \
    optim.lr=5e-5 \
    time_conditioning=False \
    parameterization=subs \
    mask_vocab_size=128 \
    model.length=256 \
    trainer.num_nodes=2 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=1024 \
    checkpointing.resume_ckpt_path=/nfs/mtr/code/ddit-c2i/outputs/old-ddit-maskgit-m128-10crop-resume1e4/05-07-090914/checkpoints/best.ckpt \

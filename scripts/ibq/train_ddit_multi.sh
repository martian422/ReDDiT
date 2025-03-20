#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MASTER_ADDR="10.10.40.232" # set the main machine with this.
MASTER_PORT=11456
NNODES=2
NPROC_PER_NODE=8
NODE_RANK=$1


ulimit -n 65536

export TORCH_DISTRIBUTED_DEBUG=DETAIL

export GLOO_SOCKET_IFNAME=bond0.45 # change this to the same as master_addr (such as eth0)
export NCCL_SOCKET_IFNAME=bond0.45 # change this to the same as master_addr (such as eth0)
export NCCL_DEBUG=INFO  # Enable debugging logs


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    model=ddit-XL \
    mode=train \
    lm_vocab_size=16384 \
    data=IBQ-token \
    wandb.name=ibq-test \
    lr_scheduler=cosine_decay_warmup_old \
    noise=loglinear \
    carry_over=True \
    rope=2d \
    repa_loss.use_repa=False \
    optim.lr=6e-4 \
    time_conditioning=False \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    trainer.num_nodes=2 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=1024 \

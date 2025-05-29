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
    model=maskgit \
    mode=train \
    lm_vocab_size=16384 \
    data=llamaGen-both-adm \
    wandb.name=gitmodel-m1 \
    lr_scheduler=cosine_decay_warmup\
    noise=loglinear \
    carry_over=True \
    rope=2d \
    repa_loss.use_repa=True \
    optim.lr=2e-4 \
    time_conditioning=False \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=1024 \
    checkpointing.resume_ckpt_path=/nfs/mtr/code/ddit-c2i/outputs/gitmodel-m1/05-27-050244/checkpoints/last.ckpt \

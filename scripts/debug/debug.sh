#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i


ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=1 \
    --master_port=11451 \
    main.py \
    model=ddit-L \
    mode=debug \
    data=llamaGen-both \
    lr_scheduler=cosine_decay_warmup \
    noise=loglinear \
    rope=1d \
    carry_over=True \
    repa_loss.use_repa=True \
    repa_loss.latent_size=16 \
    optim.lr=5e-4 \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=8 \
    loader.global_batch_size=512 \

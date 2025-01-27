#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

TOKENIZERS_PARALLELISM=false

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=1 \
    --master_port=11451 \
    main.py \
    model=L-model \
    mode=debug \
    data=llamaGen-both \
    data.dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256  \
    data.val_dataset_path=/nfs/mtr/datasets/dataset_files/imagenet-10crop-256  \
    data.image_token_dir=/nfs/mtr/datasets/imagenet_10crop_code \
    data.image_dir=/nfs/mtr/datasets/imagenet_10crops_x256 \
    data.cache_dir=/data/dataset_collects/cache \
    lr_scheduler=cosine_decay_warmup \
    noise=loglinear \
    carry_over=True \
    repa_loss.use_repa=True \
    optim.lr=5e-4 \
    data.size=1M \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=100 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=8 \
    loader.global_batch_size=512 \

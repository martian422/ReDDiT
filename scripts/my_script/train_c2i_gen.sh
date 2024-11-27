#!/bin/bash
set -x

export PYTHONPATH=$PYTHONPATH:/home/node237/Code/mdlm-c2i

WORLD_SIZE=1
RANK=0

TOKENIZERS_PARALLELISM=false

ulimit -n 65536

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=8 \
    --master_port=11456 \
    main.py \
    model=L-model-classic \
    data=llamaGen \
    data.dataset_path=/home/node237/dataset_files/imagenet-gen-cfgmix-1-2  \
    data.val_dataset_path=/home/node237/dataset_files/imagenet-gen-cfgmix-1-2  \
    data.image_token_dir=/home/node237/Workspace/Datasets/imagenet-1k/gen \
    data.cache_dir=/home/node237/dataset_files/cache \
    wandb.name=c2i-repa-ac-genmix-L-bs512 \
    lr_scheduler=cosine_decay_warmup \
    optim.lr=4e-4 \
    data.size=1M \
    generation_cfg=3.0 \
    ar_cfg=False \
    parameterization=subs \
    mask_vocab_size=1 \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=100 \
    trainer.num_nodes=1 \
    loader.num_workers=64 \
    loader.batch_size=64 \
    loader.global_batch_size=512 \

#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export HF_ENDPOINT=https://hf-mirror.com

export PYTHONPATH=$PYTHONPATH:/home/MaTianren/Workspace/MDLM-neo

RANDOM_STRING=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 8)

ulimit -n 65536

CUDA_VISIBLE_DEVICES=1,5,6,7 torchrun \
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=4 \
    --master_port=11451 \
    main.py \
    model=maskgit \
    data=llamaGen \
    data.cache_dir=/home/MaTianren/Workspace/llamaGen/dataset_files/dataset_v1 \
    parameterization=subs \
    random_noise=True \
    model.length=256 \
    eval.compute_generative_perplexity=False \
    sampling.steps=1000 \
    trainer.num_nodes=1 \
    loader.num_workers=32 \
    loader.batch_size=16 \
    loader.global_batch_size=256\
    wandb.name=mdlm-1B-llamaGen-1-BVmDXmat \
    wandb.id=mdlm-1B-llamaGen-1-BVmDXmat_1 \
    checkpointing.resume_ckpt_path=/home/MaTianren/Workspace/MDLM-neo/outputs/llamaGen/2024.09.26/110157/checkpoints/last.ckpt

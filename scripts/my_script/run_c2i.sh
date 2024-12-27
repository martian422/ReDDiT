#!/bin/bash
set -x

export WANDB_DISABLED=true

export PYTHONPATH=$PYTHONPATH:/home/node237/Code/ddit-c2i

MODEL_PATH=/home/node237/Code/ddit-c2i/outputs/c2i-dditrepa-L-m1-s500-bs512/2024.12.05/203015/checkpoints/35-90000.ckpt

CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes=1 --node-rank=0 --nproc_per_node=1 \
    --master_port=11459 \
    batch_inference.py \
    mode=sample_eval \
    model=L-model \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=2.0 \
    ar_cfg=False \
    loader.eval_batch_size=1 \
    eval.mark=None-cfg2-stable \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 

#!/bin/bash
set -x

WORLD_SIZE=1
RANK=0

export HF_ENDPOINT=https://hf-mirror.com

export PYTHONPATH=$PYTHONPATH:/home/MaTianren/Workspace/MDLM-neo

export WANDB_MODE=offline


MODEL_PATH=/home/MaTianren/Workspace/MDLM-neo/outputs/llamaGen/2024.09.27/130504/checkpoints/best.ckpt
TEXT_PROMPT="James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

CUDA_VISIBLE_DEVICES=6 torchrun\
    --nnodes=$WORLD_SIZE --node-rank=$RANK --nproc_per_node=1 \
    --master_port=11451 \
    batch_inference.py \
    mode=sample_eval \
    model=1B \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    loader.eval_batch_size=1 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=1000 \
    sampling.return_intermediate=2000 \
    sampling.num_sample_batches=1

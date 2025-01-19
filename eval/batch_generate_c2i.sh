#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/home/node237/Code/ddit-c2i

MODEL_PATH=/home/node237/Code/ddit-c2i/outputs/c2i-ddit-L-m1-d3pm/2025.01.17/215210/checkpoints/39-100000.ckpt

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=sample_eval \
    model=L-model \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=2.0 \
    ar_cfg=False \
    noise=loglinear \
    time_conditioning=True \
    loader.eval_batch_size=1 \
    eval.mark=d3pm-v1re-ddit-e40-s50-cfg2 \
    eval.mode=all \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.cfg_offset=2.0 \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 &

done

wait

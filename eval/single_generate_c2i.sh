#!/bin/bash
set -x

export WANDB_DISABLED=true
# export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:/home/node237/Code/ddit-c2i

MODEL_PATH=/home/node237/Code/ddit-c2i/outputs/c2i-ddit-L-m1-d3pm/2025.01.17/215210/checkpoints/39-100000.ckpt

CFG_SCALE=2
SAMPLE_STEP=50
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"NOBODY"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

CUDA_VISIBLE_DEVICES=0 \
    python batch_inference.py \
    mode=sample_eval \
    model=L-model \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=$CFG_SCALE \
    ar_cfg=False \
    seed=1 \
    noise=loglinear \
    time_conditioning=True \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=sample \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.cfg_offset=2.0 \
    sampling.predictor=ddpm_cache \
    sampling.steps=$SAMPLE_STEP \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 \

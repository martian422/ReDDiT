#!/bin/bash
set -x

export WANDB_DISABLED=true
# export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MODEL_PATH=/nfs/mtr/code/ddit-c2i/outputs/new-ddit-L-bs1024-m1-1d-baseline/03-20-142931/checkpoints/15-200000.ckpt

CFG_SCALE=5.0
SAMPLE_STEP=30
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"NOBODY"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

CUDA_VISIBLE_DEVICES=0 \
    python batch_inference.py \
    mode=eval \
    vq=llamagen \
    model=ddit-L \
    model.length=256 \
    lm_vocab_size=16384 \
    backbone=dit \
    data=llamaGen-both \
    mask_vocab_size=1 \
    generation_cfg=$CFG_SCALE \
    rope=1d \
    seed=1 \
    noise=loglinear \
    time_conditioning=False \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=sample \
    eval.timeline=arccos \
    eval.checkpoint_path=$MODEL_PATH \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.cfg_offset=1.5 \
    sampling.predictor=ddpm \
    sampling.steps=$SAMPLE_STEP \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 \

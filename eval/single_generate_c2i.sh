#!/bin/bash
set -x

export WANDB_DISABLED=true
# export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MODEL_PATH=/nfs/mtr/code/ddit-c2i/outputs/gitmodel-m1-extend/05-29-161759/checkpoints/19-50000.ckpt

CFG_SCALE=2.25
SAMPLE_STEP=20
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"NOBODY"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

CUDA_VISIBLE_DEVICES=0 \
    python batch_inference.py \
    mode=eval \
    vq=llamagen \
    model=maskgit \
    model.length=256 \
    lm_vocab_size=16384 \
    backbone=dit \
    mask_vocab_size=1 \
    generation_cfg=$CFG_SCALE \
    rope=2d \
    seed=1 \
    noise=cosine \
    time_conditioning=False \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=sample \
    eval.timeline=cosine \
    eval.checkpoint_path=$MODEL_PATH \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.predictor=ddpm \
    sampling.cfg_offset=2.0 \
    sampling.steps=$SAMPLE_STEP \
    sampling.num_sample_batches=1 \

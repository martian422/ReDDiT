#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MODEL_PATH=/nfs/mtr/code/ddit-c2i/outputs/mask-multi-head-2dp-repa8/02-08-114115/checkpoints/3-100000.ckpt

CFG_SCALE=2.0
SAMPLE_STEP=50
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"NOBODY"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=sample_eval \
    model=L-multi-head \
    model.length=256 \
    backbone=dit \
    data=llamaGen-token \
    mask_vocab_size=1 \
    generation_cfg=$CFG_SCALE \
    ar_cfg=False \
    mode=eval \
    rope=2d \
    seed=$GPU_ID \
    noise=loglinear \
    time_conditioning=True \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=all \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.cfg_offset=2.0 \
    sampling.predictor=ddpm_cache \
    sampling.steps=$SAMPLE_STEP \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 &

done

wait

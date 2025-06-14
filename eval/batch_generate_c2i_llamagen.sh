#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MODEL_PATH=/nfs/mtr/code/ddit-c2i/outputs/gitmodel-m128-extend/05-28-103159/checkpoints/19-50000.ckpt

CFG_SCALE=5.0
SAMPLE_STEP=20
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"NOBODY"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=eval \
    vq=llamagen \
    model=maskgit \
    model.length=256 \
    backbone=dit \
    lm_vocab_size=16384 \
    mask_vocab_size=128 \
    generation_cfg=$CFG_SCALE \
    rope=2d \
    seed=$GPU_ID \
    noise=loglinear \
    time_conditioning=False \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=all \
    eval.timeline=cosine \
    eval.checkpoint_path=$MODEL_PATH \
    eval.disable_ema=True \
    sampling.cfg_schedule=biaslinear \
    sampling.cfg_offset=1.5 \
    sampling.predictor=ddpm \
    sampling.steps=$SAMPLE_STEP \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 &

done

wait

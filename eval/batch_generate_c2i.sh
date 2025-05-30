#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/nfs/mtr/code/ddit-c2i

MODEL_PATH=/nfs/mtr/code/ddit-c2i/outputs/maskgit-ddit-cosine-norepa-1d/02-23-163448/checkpoints/13-340000.ckpt

CFG_SCALE=2.25
SAMPLE_STEP=15
EPOCH=$(echo "$MODEL_PATH" | sed -E 's#.*/([^/]+)-.*#\1#')

NAME=${1:-"eval"}

echo "evaluating model nickname: $NAME"
echo "current sampling step: $SAMPLE_STEP, with cfg = $CFG_SCALE at epoch $EPOCH."

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=eval \
    vq=maskgit \
    model=maskgit \
    model.length=256 \
    backbone=dit \
    lm_vocab_size=1024 \
    data=llamaGen-token \
    mask_vocab_size=1 \
    generation_cfg=$CFG_SCALE \
    rope=1d \
    seed=$GPU_ID \
    noise=cosine \
    time_conditioning=True \
    loader.eval_batch_size=1 \
    eval.mark=$NAME-e$EPOCH-s$SAMPLE_STEP-cfg$CFG_SCALE \
    eval.mode=all \
    eval.timeline=linear \
    eval.checkpoint_path=$MODEL_PATH \
    eval.disable_ema=True \
    sampling.cfg_schedule=const \
    sampling.cfg_offset=2.0 \
    sampling.predictor=ddpm \
    sampling.steps=$SAMPLE_STEP \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 &

done

wait

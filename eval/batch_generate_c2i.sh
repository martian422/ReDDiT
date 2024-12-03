#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/home/node237/Code/mdlm-c2i

MODEL_PATH=/home/node237/Code/mdlm-c2i/outputs/c2i-repa-ac-genmix-L-bs512/2024.11.26/214838/checkpoints/last.ckpt

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=sample_eval \
    model=L-model-classic \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=2.0 \
    ar_cfg=False \
    loader.eval_batch_size=1 \
    eval.mark=repa-s50 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1 &

done

wait

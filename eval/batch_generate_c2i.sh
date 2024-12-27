#!/bin/bash
set -x

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/home/node237/Code/ddit-c2i

MODEL_PATH=/home/node237/Code/ddit-c2i/outputs/c2i-dditrepa-L-m1-s500-bs512/2024.12.05/203015/checkpoints/107-270000.ckpt

for GPU_ID in {0..7}; do
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python batch_inference.py \
    mode=sample_eval \
    model=L-model \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=2.5 \
    ar_cfg=False \
    loader.eval_batch_size=1 \
    eval.mark=ddit-e107-s100-const25 \
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

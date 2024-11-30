TOKENIZERS_PARALLELISM=false

export WANDB_DISABLED=true
export PYTHONPATH=$PYTHONPATH:/home/node237/Code/mdlm-c2i

MODEL_PATH=/home/node237/Code/mdlm-c2i/outputs/c2i-genmix-24b16h-m1-4e4-bs512/2024.11.20/200126/checkpoints/last.ckpt
TEXT_PROMPT="1"

CUDA_VISIBLE_DEVICES=0 python batch_inference.py \
    mode=sample_eval \
    model=L-model-classic \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1 \
    generation_cfg=2.0 \
    ar_cfg=False \
    loader.eval_batch_size=1 \
    eval.mark=real-1130 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1

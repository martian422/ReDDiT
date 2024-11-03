TOKENIZERS_PARALLELISM=false

export WANDB_DISABLED=true

MODEL_PATH=/workspace/intern/liaomingxiang/ARG-MDM/MDM-1010/outputs/gen-1B-12M-m1024-s100-re/2024.10.31/190218/checkpoints/5-120000.ckpt
TEXT_PROMPT="A photo of a smiling person with snow goggles on holding a snowboard."

CUDA_VISIBLE_DEVICES=1 python inference.py \
    mode=sample_eval \
    model=1B \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1024 \
    generation_cfg=2.0 \
    ar_cfg=False \
    loader.eval_batch_size=1 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1

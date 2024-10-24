TOKENIZERS_PARALLELISM=false

export WANDB_DISABLED=true

MODEL_PATH=/workspace/intern/liaomingxiang/ARG-MDM/MDM-1010/outputs/mdm-1B-coco-5M-m1024-s24-bs512/2024.10.21/214430/checkpoints/5-50000.ckpt
TEXT_PROMPT="The coach small wrist bag is in pink."

CUDA_VISIBLE_DEVICES=1 python inference.py \
    mode=sample_eval \
    model=1B \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1024 \
    loader.eval_batch_size=1 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=10 \
    sampling.return_intermediate=0 \
    sampling.num_sample_batches=1

TOKENIZERS_PARALLELISM=false

export WANDB_DISABLED=true

MODEL_PATH=/workspace/intern/liaomingxiang/ARG-MDM/MDM-1010/outputs/neo-mdm-1B-coco-14M-m1024/2024.10.25/225608/checkpoints/1-40000.ckpt
TEXT_PROMPT="A woman in floral dress and denim jacket walking down the street."

CUDA_VISIBLE_DEVICES=1 python inference.py \
    mode=sample_eval \
    model=1B \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    mask_vocab_size=1024 \
    generation_cfg=1.0 \
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

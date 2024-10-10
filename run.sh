export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/share/project/zxs/datasets/huggingface
export TORCH_HOME=/share/project/zxs/.cache/torch

MODEL_PATH=/share/project/zxs/project/mdlm/outputs/gsm8k/2024.08.19/144923/checkpoints/833-25000.ckpt
TEXT_PROMPT="James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

python main.py \
    mode=sample_eval \
    model=1B \
    model.length=512 \
    backbone=dit \
    data=gsm8k \
    loader.eval_batch_size=1 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=10000 \
    sampling.return_intermediate=2000 \
    sampling.num_sample_batches=2

export HF_ENDPOINT=https://hf-mirror.com

export WANDB_DISABLED=true

export PYTHONPATH=$PYTHONPATH:/home/MaTianren/Workspace/MDLM-neo

MODEL_PATH=/home/MaTianren/Workspace/MDLM-neo/outputs/llamaGen/2024.10.07/224142/checkpoints/last.ckpt
TEXT_PROMPT="James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

python inference.py \
    mode=sample_eval \
    model=L-model \
    model.length=256 \
    backbone=dit \
    data=llamaGen \
    loader.eval_batch_size=1 \
    eval.checkpoint_path=$MODEL_PATH \
    eval.compute_generative_perplexity=False \
    eval.disable_ema=True \
    sampling.input_str="$TEXT_PROMPT" \
    sampling.predictor=ddpm_cache \
    sampling.steps=10000 \
    sampling.return_intermediate=2000 \
    sampling.num_sample_batches=1

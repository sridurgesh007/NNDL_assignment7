#!/usr/bin/env bash
#SBATCH --job-name=llama2_lora_dolly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00


module purge
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llm

# Hugging Face token
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"


export WANDB_API_KEY=""
export WANDB_MODE=offline
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

mkdir -p logs outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install --upgrade pip wheel
pip uninstall -y accelerate || true

pip install "accelerate>=0.34.2,<1.0.0" "transformers>=4.57.1" \
            "trl==0.18.2" "peft>=0.12.0" "bitsandbytes==0.42.0" "datasets>=2.19.0" \
            "wandb"

export PYTHONNOUSERSITE=1


python - <<'PY'
import sys, inspect
import accelerate, transformers
from accelerate.accelerator import Accelerator
print("PY:", sys.executable)
print("accelerate:", accelerate.__version__, accelerate.__file__)
print("transformers:", transformers.__version__)
print("unwrap sig:", inspect.signature(Accelerator.unwrap_model))
PY

echo "[W&B] key length: $(echo -n "${WANDB_API_KEY:-}" | wc -c)"
echo "[W&B] entity=${WANDB_ENTITY:-<none>} mode=${WANDB_MODE:-online}"
python - <<'PY'
import os, sys
print("[PY]", sys.executable)
print("[WANDB] present:", bool(os.environ.get("WANDB_API_KEY")))
PY
# --- Run ---
python -u train2.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name databricks/databricks-dolly-15k \
  --output_dir outputs4/llama2-7b-dolly15k-lora \
  --epochs 6 \
  --batch_size 4 \
  --grad_accum 16 \
  --max_seq_length 2048 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_dropout 0.01 \
  --weight_decay 0.1 \
  --bf16 true \
  --fp16 false \
  --qlora true \
  --packing \
  --logging_steps 5 \
  --eval_steps 5 \
  --save_steps 600 \
  --save_total_limit 2 \
  --merge_and_save true \
  --wandb_project "llama2-finetune-dolly" \
  --wandb_run_name "r64-alpha128-lr1e-4-wd0.05-ep4-ctx2048"

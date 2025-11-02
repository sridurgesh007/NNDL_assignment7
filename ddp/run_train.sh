#!/usr/bin/env bash
#SBATCH --job-name=llama2_lora_dolly_ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00



# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate llm

source /home/s/sri007/my_project/assignment7/llama_ddp/bin/activate

echo "Running Python script for answer generation..."
# Hugging Face token
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"


export WANDB_API_KEY="8994bfd3ba326df66ddc6c07daae079a2052257b"
export WANDB_MODE=offline
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# DDP environment variables
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

mkdir -p logs outputs


pip install --progress-bar off --upgrade pip wheel
pip install "datasets<4.0"
pip install --upgrade accelerate
pip install -U bitsandbytes
pip install --no-progress-bar --upgrade accelerate peft transformers
pip install --no-progress-bar bitsandbytes
export PYTHONNOUSERSITE=1

# Simple verification
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("PyTorch version:", torch.__version__)
PY

echo "[W&B] key length: $(echo -n "${WANDB_API_KEY:-}" | wc -c)"


python train2.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name databricks/databricks-dolly-15k \
  --output_dir outputs6/llama2-7b-dolly15k-lora-ddp \
  --epochs 3 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_seq_length 2048 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --weight_decay 0.01 \
  --bf16 true \
  --fp16 false \
  --qlora true \
  --packing \
  --logging_steps 5 \
  --eval_steps 5 \
  --save_steps 500 \
  --save_total_limit 3 \
  --merge_and_save true \
  --wandb_project "llama2-finetune-dolly" \
  --wandb_run_name "r32-alpha64-lr2e-4-wd0.01-ep3-ctx2048"
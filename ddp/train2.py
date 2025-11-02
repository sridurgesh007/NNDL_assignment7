#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-GPU PEFT-LoRA fine-tuning with SFTTrainer (DDP compatible)
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional
import wandb
# plotting (non-interactive)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig


# ---------- prompt templates ----------
TEMPLATE_WITH_CTX = (
    "### Instruction:\n{instruction}\n\n"
    "### Context:\n{context}\n\n"
    "### Response:\n{response}"
)
TEMPLATE_NO_CTX = (
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}"
)

def format_entry(example: Dict) -> Dict:
    instr = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    resp = (example.get("response") or "").strip()
    if len(resp) == 0:
        return {"text": ""}
    if len(ctx) > 0:
        return {"text": TEMPLATE_WITH_CTX.format(instruction=instr, context=ctx, response=resp)}
    else:
        return {"text": TEMPLATE_NO_CTX.format(instruction=instr, response=resp)}

def get_bnb_config(load_in_4bit: bool, load_in_8bit: bool) -> Optional[BitsAndBytesConfig]:
    if not (load_in_4bit or load_in_8bit):
        return None
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None

def make_lora_config(r: int, alpha: int, dropout: float, target_modules: Optional[List[str]] = None) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        task_type=TaskType.CAUSAL_LM, target_modules=target_modules, bias="none"
    )

def write_args_card(args: argparse.Namespace, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

def dump_logs_to_csv(trainer_state: dict, out_dir: str):
    """Write train/eval losses into CSVs and make a PNG curve."""
    os.makedirs(out_dir, exist_ok=True)
    log_history = trainer_state.get("log_history", [])
    train_rows, eval_rows = [], []
    for item in log_history:
        step = item.get("step", None)
        if "loss" in item:
            train_rows.append({"step": step, "loss": item["loss"]})
        if "eval_loss" in item:
            eval_rows.append({"step": step, "eval_loss": item["eval_loss"]})

    with open(os.path.join(out_dir, "train_loss.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss"]); w.writeheader()
        w.writerows(train_rows)
    with open(os.path.join(out_dir, "eval_loss.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "eval_loss"]); w.writeheader()
        w.writerows(eval_rows)

    plt.figure()
    if train_rows:
        plt.plot([r["step"] for r in train_rows], [r["loss"] for r in train_rows], label="train")
    if eval_rows:
        plt.plot([r["step"] for r in eval_rows], [r["eval_loss"] for r in eval_rows], label="eval")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training / Evaluation Loss")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png")); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset_name", type=str, default="databricks/databricks-dolly-15k")
    parser.add_argument("--output_dir", type=str, default="./outputs/llama2-7b-dolly15k-lora")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--packing", action="store_true")

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--qlora", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--load_in_8bit", type=lambda x: str(x).lower() == "true", default=False)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=True)

    parser.add_argument("--merge_and_save", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name .")
    args = parser.parse_args()

    # --------- setup ----------
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    write_args_card(args, args.output_dir)

    use_wandb = bool(args.wandb_project)
    
    if use_wandb:
        print(f"[INFO] W&B Project specified: {args.wandb_project}. Attempting W&B setup.")
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
             os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
        
        try:
            _tmp = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                settings=wandb.Settings(start_method="thread"),
                reinit=True,
            )
            print("[W&B] Preflight OK →", _tmp.get_url() if _tmp else "URL not available")
            if _tmp:
                 wandb.mark_preempting()
            report_to_value = ["wandb"]
        except Exception as e:
            print("[W&B] Preflight failed:", repr(e))
            os.environ.setdefault("WANDB_MODE", "offline")
            print("[W&B] Falling back to offline mode.")
            report_to_value = ["wandb"]
    else:
        print("[INFO] No W&B Project specified. Skipping W&B.")
        report_to_value = "none"

    # --------- data ----------
    raw = load_dataset(args.dataset_name)
    mapped = raw.map(lambda ex: format_entry(ex), remove_columns=raw["train"].column_names)
    mapped = mapped.filter(lambda x: len(x["text"]) > 0)
    if "validation" not in mapped and "test" not in mapped:
        split = mapped["train"].train_test_split(test_size=0.2, seed=args.seed)
        valtest = split["test"].train_test_split(test_size=0.5, seed=args.seed)
        dataset = {"train": split["train"], "validation": valtest["train"], "test": valtest["test"]}
    else:
        dataset = {
            "train": mapped.get("train", None),
            "validation": mapped.get("validation", mapped.get("test", None)),
            "test": mapped.get("test", None),
        }

    # --------- tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # --------- model + PEFT (QLoRA) ----------
    print("Getting bnb config")
    bnb_config = get_bnb_config(load_in_4bit=args.qlora, load_in_8bit=args.load_in_8bit)
    print(bnb_config)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        trust_remote_code=False,
        
    )
    
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.qlora or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_cfg = make_lora_config(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --------- TRL SFT config ----------
    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to_value, 
        run_name=args.wandb_run_name if use_wandb else None,
        bf16=args.bf16,
        fp16=args.fp16,
        eval_accumulation_steps=2,
        dataloader_num_workers=4,
        dataloader_drop_last=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_8bit" if (args.qlora or args.load_in_8bit) else "adamw_torch",
        lr_scheduler_type="cosine_with_restarts",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        ddp_find_unused_parameters=False,  # Important for DDP stability
        ddp_timeout=1800,  
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator,
        processing_class=tokenizer,
    )

    # --------- train ----------
    print("Training started")
    train_result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)

    # --------- evaluate & save metrics ----------
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss", None)
    if eval_loss is not None and eval_loss > 0:
        metrics["eval_perplexity"] = math.exp(eval_loss)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    
    state_path = os.path.join(args.output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        dump_logs_to_csv(state, args.output_dir)

    if args.merge_and_save:
        print("Merging LoRA adapters into base model and saving full weights...")
        merged = AutoPeftModelForCausalLM.from_pretrained(args.output_dir, torch_dtype=torch.bfloat16)
        merged = merged.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "merged")
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        
    if use_wandb and os.environ.get("WANDB_MODE", "") != "offline":
         wandb.finish()
    
    print("Done.")

if __name__ == "__main__":
    main()
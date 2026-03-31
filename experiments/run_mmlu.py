"""
experiments/run_mmlu.py
-----------------------
MMLU benchmarking: FNA vs LoRA on T5-small.

Usage (Colab):
    # FNA run
    !python experiments/run_mmlu.py --adapter fna --tasks anatomy,astronomy --epochs 5

    # LoRA baseline
    !python experiments/run_mmlu.py --adapter lora --tasks anatomy,astronomy --epochs 5

    # Compare
    !python experiments/compare.py

Setup (run once):
    !pip install transformers datasets accelerate sentencepiece peft -q

Author: Shiva Shankar Kanike
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent dir to path so fna package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from fna import inject_fna_adapters, get_fna_optimizer_params, FNAOptimizer, print_model_summary

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME  = "t5-small"
MAX_INPUT   = 512
MAX_TARGET  = 4
BATCH_SIZE  = 16
GRID_SIZE   = 128   # FNA latent grid (16,384 params per layer)
ALPHA       = 1.0   # DeltaW scaling
LORA_RANK   = 8     # LoRA rank for baseline (~20,480 params per layer)

MMLU_TASKS = [
    "anatomy", "astronomy", "college_mathematics",
    "high_school_physics", "moral_scenarios",
    "professional_law", "world_religions", "global_facts",
]

# FNA hyperparameters
FNA_DT = 0.05
FNA_NU = 0.10

# LoRA hyperparameters
LORA_LR = 3e-4


# ---------------------------------------------------------------------------
# MMLU data utilities
# ---------------------------------------------------------------------------

def format_mmlu(example):
    choices    = example["choices"]
    labels     = ["A", "B", "C", "D"]
    choices_str = " ".join(f"{l}. {c}" for l, c in zip(labels, choices))
    input_text  = f"question: {example['question']} choices: {choices_str}"
    target_text = labels[example["answer"]]
    return input_text, target_text


def make_loader(task, tokenizer, split="test", batch_size=BATCH_SIZE):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", task, split=split, trust_remote_code=False)

    inputs, targets = [], []
    for ex in ds:
        inp, tgt = format_mmlu(ex)
        inputs.append(inp)
        targets.append(tgt)

    enc = tokenizer(
        inputs, truncation=True, max_length=MAX_INPUT,
        padding="max_length", return_tensors="pt",
    )
    lbl = tokenizer(
        targets, max_length=MAX_TARGET,
        padding="max_length", return_tensors="pt",
    ).input_ids
    lbl[lbl == tokenizer.pad_token_id] = -100

    dataset = torch.utils.data.TensorDataset(
        enc.input_ids, enc.attention_mask, lbl
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split != "test"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, task, device):
    loader = make_loader(task, tokenizer, split="test", batch_size=32)
    model.eval()

    label_ids = {
        lbl: tokenizer.encode(lbl, add_special_tokens=False)[0]
        for lbl in ["A", "B", "C", "D"]
    }

    correct, total = 0, 0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        outputs  = model.generate(
            input_ids=input_ids, attention_mask=attn_mask,
            max_new_tokens=1, do_sample=False,
        )
        pred_ids = outputs[:, 1] if outputs.shape[1] > 1 else outputs[:, 0]

        for pred_id, label_row in zip(pred_ids, labels):
            gold = label_row[label_row != -100]
            if len(gold) == 0:
                continue
            if pred_id.item() == gold[0].item():
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# FNA training
# ---------------------------------------------------------------------------

def train_fna(tasks, epochs, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"Adapter: FNA (full NS)  |  Device: {device}")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}={GRID_SIZE*GRID_SIZE:,} params/layer")
    print(f"NS params: dt={FNA_DT}, nu={FNA_NU}")
    print(f"{'='*55}\n")

    print("Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Inject FNA adapters into FFN layers
    fna_layers = inject_fna_adapters(
        model,
        target_modules=["DenseReluDense.wi", "DenseReluDense.wo"],
        grid_size=GRID_SIZE,
        alpha=ALPHA,
        verbose=True,
    )
    model = model.to(device)
    print_model_summary(model)

    # NS optimizer only for trainable M grids
    fna_params = get_fna_optimizer_params(model)
    optimizer  = FNAOptimizer(fna_params, dt=FNA_DT, nu=FNA_NU)

    results = {
        "adapter":    "fna",
        "tasks":      tasks,
        "grid_size":  GRID_SIZE,
        "dt":         FNA_DT,
        "nu":         FNA_NU,
        "trainable_params": sum(p.numel() for p in fna_params),
        "per_task":   {},
    }

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        try:
            loader = make_loader(task, tokenizer, split="validation")
        except Exception:
            loader = make_loader(task, tokenizer, split="test")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, n = 0.0, 0

            for input_ids, attn_mask, labels in loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)

                optimizer.zero_grad()
                out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n += 1

            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(n,1):.4f}")

        acc     = evaluate(model, tokenizer, task, device)
        elapsed = time.time() - t0
        print(f"  Accuracy: {acc*100:.2f}%  ({elapsed:.1f}s)")
        results["per_task"][task] = {"accuracy": round(acc, 4), "time_s": round(elapsed, 1)}

        # Reset velocity between tasks so momentum doesn't bleed over
        optimizer.zero_velocity()

    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs) / len(accs), 4)
    print(f"\nMean accuracy: {results['mean_accuracy']*100:.2f}%")

    os.makedirs(results_dir, exist_ok=True)
    out_path = Path(results_dir) / "fna_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# LoRA baseline training
# ---------------------------------------------------------------------------

def train_lora(tasks, epochs, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"Adapter: LoRA r={LORA_RANK}  |  Device: {device}")
    print(f"{'='*55}\n")

    print("Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=["q", "v"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    results = {
        "adapter":    "lora",
        "tasks":      tasks,
        "rank":       LORA_RANK,
        "lr":         LORA_LR,
        "trainable_params": sum(p.numel() for p in lora_params),
        "per_task":   {},
    }

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        try:
            loader = make_loader(task, tokenizer, split="validation")
        except Exception:
            loader = make_loader(task, tokenizer, split="test")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, n = 0.0, 0

            for input_ids, attn_mask, labels in loader:
                input_ids = input_ids.to(device)
                attn_mask = attn_mask.to(device)
                labels    = labels.to(device)

                optimizer.zero_grad()
                out  = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n += 1

            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(n,1):.4f}")

        acc     = evaluate(model, tokenizer, task, device)
        elapsed = time.time() - t0
        print(f"  Accuracy: {acc*100:.2f}%  ({elapsed:.1f}s)")
        results["per_task"][task] = {"accuracy": round(acc, 4), "time_s": round(elapsed, 1)}

    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs) / len(accs), 4)
    print(f"\nMean accuracy: {results['mean_accuracy']*100:.2f}%")

    os.makedirs(results_dir, exist_ok=True)
    out_path = Path(results_dir) / "lora_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNA vs LoRA on MMLU")
    parser.add_argument("--adapter", choices=["fna", "lora"], default="fna")
    parser.add_argument("--tasks",   type=str, default="anatomy,astronomy",
                        help="Comma-separated MMLU task names")
    parser.add_argument("--epochs",  type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.adapter == "fna":
        train_fna(tasks, args.epochs, args.results_dir)
    else:
        train_lora(tasks, args.epochs, args.results_dir)

"""
experiments/needle.py
---------------------
Needle-in-a-Haystack experiment.

Tests whether FNA's pressure term suppresses irrelevant context noise
better than LoRA and the base model.

Setup:
    - Hide a simple fact (the "needle") in a passage of random distractor
      sentences (the "haystack")
    - Ask the model to retrieve the fact
    - Vary the noise ratio (0% to 90% distractors)
    - Plot accuracy vs noise ratio for: Base, LoRA, FNA, FNA+Memory

Example needle:
    "The capital of France is Paris."
    Question: "What is the capital of France?"
    Answer: "Paris"

Distractors: random factual sentences that are irrelevant to the question.

Usage:
    !python experiments/needle.py --model t5-small --n_seeds 3

Author: Shiva Shankar Kanike
"""

import argparse
import json
import os
import sys
import random
import time
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from fna import (inject_fna_adapters, get_fna_optimizer_params,
                 FNAOptimizer, FNAMemoryLayer, inject_fna_memory)

MODEL_NAME = "t5-small"
LORA_RANK  = 16
GRID_SIZE  = 128
FNA_LR     = 3e-4
FNA_NU     = 0.2
LORA_LR    = 3e-4

ENCODER_QV = [
    "encoder.block.0.layer.0.SelfAttention.q",
    "encoder.block.0.layer.0.SelfAttention.v",
    "encoder.block.1.layer.0.SelfAttention.q",
    "encoder.block.1.layer.0.SelfAttention.v",
    "encoder.block.2.layer.0.SelfAttention.q",
    "encoder.block.2.layer.0.SelfAttention.v",
    "encoder.block.3.layer.0.SelfAttention.q",
    "encoder.block.3.layer.0.SelfAttention.v",
    "encoder.block.4.layer.0.SelfAttention.q",
    "encoder.block.4.layer.0.SelfAttention.v",
    "encoder.block.5.layer.0.SelfAttention.q",
    "encoder.block.5.layer.0.SelfAttention.v",
]

# ---------------------------------------------------------------------------
# Needle-haystack data
# ---------------------------------------------------------------------------

# Each needle is (fact, question, answer)
NEEDLES = [
    ("The access code is A1-X9.", "What is the access code?", "A1-X9"),
    ("The secret key is Alpha-7.", "What is the secret key?", "Alpha-7"),
    ("The project ID is BR-442.", "What is the project ID?", "BR-442"),
    ("The batch number is CM-991.", "What is the batch number?", "CM-991"),
    ("The file code is DX-115.", "What is the file code?", "DX-115"),
    ("The reference ID is EV-338.", "What is the reference ID?", "EV-338"),
    ("The serial number is FZ-774.", "What is the serial number?", "FZ-774"),
    ("The ticket code is GK-221.", "What is the ticket code?", "GK-221"),
    ("The entry key is HT-559.", "What is the entry key?", "HT-559"),
    ("The system ID is IQ-883.", "What is the system ID?", "IQ-883"),
    ("The access code is JM-116.", "What is the access code?", "JM-116"),
    ("The secret key is KP-447.", "What is the secret key?", "KP-447"),
    ("The project ID is LR-772.", "What is the project ID?", "LR-772"),
    ("The batch number is MS-338.", "What is the batch number?", "MS-338"),
    ("The file code is NT-664.", "What is the file code?", "NT-664"),
    ("The reference ID is OV-991.", "What is the reference ID?", "OV-991"),
    ("The serial number is PW-225.", "What is the serial number?", "PW-225"),
    ("The ticket code is QX-558.", "What is the ticket code?", "QX-558"),
    ("The entry key is RY-884.", "What is the entry key?", "RY-884"),
    ("The system ID is SZ-117.", "What is the system ID?", "SZ-117"),
    ("The access code is TA-443.", "What is the access code?", "TA-443"),
    ("The secret key is UB-776.", "What is the secret key?", "UB-776"),
    ("The project ID is VC-112.", "What is the project ID?", "VC-112"),
    ("The batch number is WD-448.", "What is the batch number?", "WD-448"),
    ("The file code is XE-773.", "What is the file code?", "XE-773"),
    ("The reference ID is YF-339.", "What is the reference ID?", "YF-339"),
    ("The serial number is ZG-665.", "What is the serial number?", "ZG-665"),
    ("The ticket code is AH-992.", "What is the ticket code?", "AH-992"),
    ("The entry key is BI-226.", "What is the entry key?", "BI-226"),
    ("The system ID is CJ-559.", "What is the system ID?", "CJ-559"),
    ("The access code is DK-885.", "What is the access code?", "DK-885"),
    ("The secret key is EL-118.", "What is the secret key?", "EL-118"),
    ("The project ID is FM-444.", "What is the project ID?", "FM-444"),
    ("The batch number is GN-779.", "What is the batch number?", "GN-779"),
    ("The file code is HO-113.", "What is the file code?", "HO-113"),
    ("The reference ID is IP-446.", "What is the reference ID?", "IP-446"),
    ("The serial number is JQ-775.", "What is the serial number?", "JQ-775"),
    ("The ticket code is KR-331.", "What is the ticket code?", "KR-331"),
    ("The entry key is LS-668.", "What is the entry key?", "LS-668"),
    ("The system ID is MT-994.", "What is the system ID?", "MT-994"),
    ("The access code is NU-228.", "What is the access code?", "NU-228"),
    ("The secret key is OV-561.", "What is the secret key?", "OV-561"),
    ("The project ID is PW-887.", "What is the project ID?", "PW-887"),
    ("The batch number is QX-121.", "What is the batch number?", "QX-121"),
    ("The file code is RY-454.", "What is the file code?", "RY-454"),
    ("The reference ID is SZ-783.", "What is the reference ID?", "SZ-783"),
    ("The serial number is TA-319.", "What is the serial number?", "TA-319"),
    ("The ticket code is UB-642.", "What is the ticket code?", "UB-642"),
    ("The entry key is VC-975.", "What is the entry key?", "VC-975"),
    ("The system ID is WD-308.", "What is the system ID?", "WD-308")
]

# Distractor sentences — irrelevant facts used as noise
DISTRACTORS = [
    "The butterfly effect refers to sensitive dependence on initial conditions.",
    "Chlorophyll gives plants their green color.",
    "The printing press was invented in the 15th century.",
    "Neurons transmit electrical signals in the brain.",
    "The Sahara is the largest hot desert in the world.",
    "Pluto was reclassified as a dwarf planet in 2006.",
    "The human genome contains approximately 3 billion base pairs.",
    "Sound travels faster through solids than through air.",
    "The Roman Empire fell in 476 AD.",
    "Penguins are found primarily in the Southern Hemisphere.",
    "The transistor was invented in 1947.",
    "Coral reefs are built by tiny marine organisms called polyps.",
    "The Renaissance began in Italy in the 14th century.",
    "Bats navigate using echolocation.",
    "The first computer bug was an actual insect found in a relay.",
    "Volcanoes form at tectonic plate boundaries.",
    "The mitochondria is the powerhouse of the cell.",
    "Light takes about 8 minutes to travel from the Sun to Earth.",
    "The periodic table was organized by Dmitri Mendeleev.",
    "Bees communicate through a waggle dance.",
    "The tallest building in the world is the Burj Khalifa.",
    "Chess originated in India around the 6th century.",
    "The ozone layer protects Earth from ultraviolet radiation.",
    "Diamonds are made of carbon atoms arranged in a crystal structure.",
    "The first moon landing occurred in July 1969.",
    "Antibiotics were discovered by Alexander Fleming.",
    "The speed of sound at sea level is about 343 meters per second.",
    "Earthquakes are measured using the Richter scale.",
    "The human eye can distinguish about 10 million colors.",
    "Whales breathe air through blowholes.",
    "The Internet was initially developed as ARPANET.",
    "Mars has two small moons named Phobos and Deimos.",
    "The Nile is the longest river in Africa.",
    "Rainbows form when sunlight refracts through water droplets.",
    "The boiling point of water decreases at higher altitudes.",
    "Cells divide through a process called mitosis.",
    "The first flight by the Wright Brothers lasted 12 seconds.",
    "Black holes have gravity so strong that light cannot escape.",
    "The human brain weighs approximately 1.4 kilograms.",
    "Tectonic plates move at roughly the same rate fingernails grow.",
]


def build_haystack(needle_fact, noise_ratio, max_tokens=400):
    """
    Build an input context with the needle buried in distractors.

    Args:
        needle_fact: the fact sentence to hide
        noise_ratio: fraction of context that is distractors (0.0 to 0.9)
        max_tokens:  approximate max context length in words

    Returns:
        context: string with needle + distractors in random order
    """
    if noise_ratio == 0.0:
        return needle_fact

    # Calculate number of distractor sentences
    # Each distractor is ~12 words on average
    words_for_noise  = int(max_tokens * noise_ratio)
    n_distractors    = max(1, words_for_noise // 12)
    n_distractors    = min(n_distractors, len(DISTRACTORS))

    selected = random.sample(DISTRACTORS, n_distractors)

    # Insert needle at a random position
    insert_pos = random.randint(0, len(selected))
    selected.insert(insert_pos, needle_fact)

    return " ".join(selected)


def format_needle_input(context, question):
    return "question: " + question + " context: " + context


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_base_model(tokenizer_only=False):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    if tokenizer_only:
        return tokenizer, None
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    return tokenizer, model


def build_fna_model(seed=42):
    import random as rnd
    rnd.seed(seed)
    torch.manual_seed(seed)
    tokenizer, model = build_base_model()
    inject_fna_adapters(model, target_modules=ENCODER_QV,
                        grid_size=GRID_SIZE, alpha=1.0, verbose=False)
    params    = get_fna_optimizer_params(model)
    optimizer = FNAOptimizer(params, lr=FNA_LR, nu=FNA_NU)
    return tokenizer, model, optimizer, params


def build_fna_memory_model(seed=42):
    import random as rnd
    rnd.seed(seed)
    torch.manual_seed(seed)
    tokenizer, model = build_base_model()
    # Inject both adapter AND memory layers
    inject_fna_adapters(model, target_modules=ENCODER_QV,
                        grid_size=GRID_SIZE, alpha=1.0, verbose=False)
    inject_fna_memory(model, d_model=512,
                      after_modules=["SelfAttention"],
                      bottleneck=16, nu_init=FNA_NU,
                      learn_nu=True, verbose=False)
    params    = get_fna_optimizer_params(model)
    optimizer = FNAOptimizer(params, lr=FNA_LR, nu=FNA_NU)
    return tokenizer, model, optimizer, params


def build_lora_model(seed=42):
    import math as _math
    import random as rnd
    rnd.seed(seed)
    torch.manual_seed(seed)

    class LoRALayer(nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
            self.bias   = (nn.Parameter(linear.bias.data.clone(), requires_grad=False)
                           if linear.bias is not None else None)
            in_f, out_f = linear.in_features, linear.out_features
            self.A = nn.Parameter(torch.empty(rank, in_f))
            self.B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.A, a=_math.sqrt(5))
        def forward(self, x):
            import torch.nn.functional as F
            base = F.linear(x, self.weight, self.bias)
            return base + F.linear(F.linear(x, self.A), self.B)

    tokenizer, model = build_base_model()
    for p in model.parameters():
        p.requires_grad = False
    for name, mod in model.named_modules():
        if name in ENCODER_QV and isinstance(mod, nn.Linear):
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALayer(mod, LORA_RANK))

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=0.01)
    return tokenizer, model, optimizer, params


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_on_needles(model, tokenizer, optimizer, params,
                     noise_ratio, device, epochs=3, is_fna=True):
    """Fine-tune model on needle retrieval at a fixed noise ratio."""
    model.train()
    model = model.to(device)

    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        random.shuffle(NEEDLES)

        for fact, question, answer in NEEDLES:
            context = build_haystack(fact, noise_ratio)
            inp     = format_needle_input(context, question)

            enc = tokenizer(inp, return_tensors="pt",
                            truncation=True, max_length=512).to(device)
            lbl = tokenizer(answer, return_tensors="pt",
                            max_length=16).input_ids.to(device)
            lbl[lbl == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            out  = model(**enc, labels=lbl)
            loss = out.loss
            loss.backward()

            if not is_fna:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_needles(model, tokenizer, noise_ratio, device, n_eval=20):
    model.eval()
    correct, total = 0, 0
    random.seed(int(noise_ratio * 100))  # different sample per noise level
    eval_needles = NEEDLES[:n_eval]

    for fact, question, answer in eval_needles:
        context = build_haystack(fact, noise_ratio)
        inp     = format_needle_input(context, question)

        enc = tokenizer(inp, return_tensors="pt",
                        truncation=True, max_length=512).to(device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip().lower()

        # Strict match: answer must appear exactly in prediction
        if answer.lower() in pred:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(noise_ratios, n_seeds, results_dir, epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Noise ratios:", noise_ratios)
    print("Seeds:", n_seeds)

    results = {
        "noise_ratios": noise_ratios,
        "epochs": epochs,
        "adapters": {
            "base":       {str(r): [] for r in noise_ratios},
            "lora":       {str(r): [] for r in noise_ratios},
            "fna":        {str(r): [] for r in noise_ratios},
            "fna_memory": {str(r): [] for r in noise_ratios},
        }
    }

    for seed in range(n_seeds):
        print("\n" + "="*50)
        print("Seed:", seed)
        print("="*50)
        random.seed(seed)
        torch.manual_seed(seed)

        for noise_ratio in noise_ratios:
            print("\nNoise ratio:", noise_ratio)

            # Base model (no adapter, no training)
            print("  Base model...")
            tokenizer, base_model = build_base_model()
            base_model = base_model.to(device)
            acc = evaluate_needles(base_model, tokenizer, noise_ratio, device)
            results["adapters"]["base"][str(noise_ratio)].append(round(acc, 4))
            print("  Base accuracy:", round(acc*100, 2))
            del base_model

            # LoRA
            print("  LoRA...")
            tokenizer, lora_model, lora_opt, lora_params = build_lora_model(seed)
            lora_model = lora_model.to(device)
            train_on_needles(lora_model, tokenizer, lora_opt, lora_params,
                             noise_ratio, device, epochs=epochs, is_fna=False)
            acc = evaluate_needles(lora_model, tokenizer, noise_ratio, device)
            results["adapters"]["lora"][str(noise_ratio)].append(round(acc, 4))
            print("  LoRA accuracy:", round(acc*100, 2))
            del lora_model

            # FNA adapter
            print("  FNA adapter...")
            tokenizer, fna_model, fna_opt, fna_params = build_fna_model(seed)
            fna_model = fna_model.to(device)
            train_on_needles(fna_model, tokenizer, fna_opt, fna_params,
                             noise_ratio, device, epochs=epochs, is_fna=True)
            acc = evaluate_needles(fna_model, tokenizer, noise_ratio, device)
            results["adapters"]["fna"][str(noise_ratio)].append(round(acc, 4))
            print("  FNA accuracy:", round(acc*100, 2))
            del fna_model

            # FNA + Memory layer
            print("  FNA + Memory...")
            tokenizer, fnam_model, fnam_opt, fnam_params = build_fna_memory_model(seed)
            fnam_model = fnam_model.to(device)
            train_on_needles(fnam_model, tokenizer, fnam_opt, fnam_params,
                             noise_ratio, device, epochs=epochs, is_fna=True)
            acc = evaluate_needles(fnam_model, tokenizer, noise_ratio, device)
            results["adapters"]["fna_memory"][str(noise_ratio)].append(round(acc, 4))
            print("  FNA+Memory accuracy:", round(acc*100, 2))
            del fnam_model

            torch.cuda.empty_cache()

    # Compute means
    import numpy as np
    summary = {"noise_ratios": noise_ratios, "means": {}}
    for adapter in results["adapters"]:
        summary["means"][adapter] = {}
        for r in noise_ratios:
            vals = results["adapters"][adapter][str(r)]
            summary["means"][adapter][str(r)] = round(float(np.mean(vals))*100, 2)

    # Print table
    print("\n" + "="*65)
    print("Needle-in-Haystack Results (mean accuracy %)")
    print("="*65)
    header = "Noise    Base      LoRA      FNA       FNA+Mem"
    print(header)
    print("-"*65)
    for r in noise_ratios:
        row = str(int(r*100)) + "%"
        for adapter in ["base", "lora", "fna", "fna_memory"]:
            row += "     " + str(summary["means"][adapter][str(r)]) + "%"
        print(row)
    print("="*65)

    # Save
    os.makedirs(results_dir, exist_ok=True)
    path = Path(results_dir) / "needle_results.json"
    with open(path, "w") as f:
        json.dump({"raw": results, "summary": summary}, f, indent=2)
    print("\nSaved to", str(path))

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_ratios", type=str,
                        default="0.0,0.2,0.4,0.6,0.8",
                        help="Comma-separated noise ratios")
    parser.add_argument("--n_seeds",     type=int, default=2)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    noise_ratios = [float(x) for x in args.noise_ratios.split(",")]

    run_experiment(
        noise_ratios=noise_ratios,
        n_seeds=args.n_seeds,
        results_dir=args.results_dir,
        epochs=args.epochs,
    )

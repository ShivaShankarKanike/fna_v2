
import argparse, json, os, sys, time, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from fna import inject_fna_adapters, get_fna_optimizer_params, FNAOptimizer

# ── Fixed config ────────────────────────────────────────────
MODEL_NAME  = "t5-small"
MAX_INPUT   = 512
MAX_TARGET  = 4
BATCH_SIZE  = 16
GRID_SIZE   = 128        # FNA: 128x128 = 16,384 params per layer
FNA_LR      = 3e-4
FNA_NU      = 0.2        # best from ablation
LORA_LR     = 3e-4

# Exactly 12 encoder q,v layers — explicit full names, no ambiguity
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
# ── End config ───────────────────────────────────────────────


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_mmlu(example):
    labels = ["A","B","C","D"]
    cs = " ".join(f"{l}. {c}" for l,c in zip(labels, example["choices"]))
    return f"question: {example['question']} choices: {cs}", labels[example["answer"]]


def make_loader(task, tokenizer, split="test", batch_size=BATCH_SIZE):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", task, split=split, trust_remote_code=False)
    inputs, targets = [], []
    for ex in ds:
        i, t = format_mmlu(ex)
        inputs.append(i)
        targets.append(t)
    enc = tokenizer(inputs, truncation=True, max_length=MAX_INPUT,
                    padding="max_length", return_tensors="pt")
    lbl = tokenizer(targets, max_length=MAX_TARGET,
                    padding="max_length", return_tensors="pt").input_ids
    lbl[lbl == tokenizer.pad_token_id] = -100
    dataset = torch.utils.data.TensorDataset(enc.input_ids, enc.attention_mask, lbl)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate(model, tokenizer, task, device):
    loader = make_loader(task, tokenizer, split="test", batch_size=32)
    model.eval()
    label_ids = {l: tokenizer.encode(l, add_special_tokens=False)[0]
                 for l in ["A","B","C","D"]}
    correct, total = 0, 0
    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                 max_new_tokens=1, do_sample=False)
        pred_ids = outputs[:,1] if outputs.shape[1] > 1 else outputs[:,0]
        for pred_id, label_row in zip(pred_ids, labels):
            gold = label_row[label_row != -100]
            if len(gold) == 0:
                continue
            if pred_id.item() == gold[0].item():
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def train_fna(tasks, epochs, lora_rank, seed, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FNA params: 12 layers x 128x128 = 196,608
    fna_params_count = len(ENCODER_QV) * GRID_SIZE * GRID_SIZE
    print(f"Adapter: FNA | layers={len(ENCODER_QV)} | grid={GRID_SIZE}x{GRID_SIZE}")
    print(f"Trainable params: {fna_params_count:,} | nu={FNA_NU} | lr={FNA_LR} | seed={seed}")

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    fna_layers = inject_fna_adapters(model, target_modules=ENCODER_QV,
                                     grid_size=GRID_SIZE, alpha=1.0, verbose=False)
    model = model.to(device)

    fna_params = get_fna_optimizer_params(model)
    actual_count = sum(p.numel() for p in fna_params)
    print(f"Actual trainable params: {actual_count:,}")

    optimizer = FNAOptimizer(fna_params, lr=FNA_LR, nu=FNA_NU)

    results = {
        "adapter": "fna", "tasks": tasks, "epochs": epochs,
        "grid_size": GRID_SIZE, "nu": FNA_NU, "lr": FNA_LR, "seed": seed,
        "trainable_params": actual_count, "per_task": {}
    }

    for task in tasks:
        print(f"\n--- Task: {task} ---")
        loader = make_loader(task, tokenizer, split="test")
        t0 = time.time()
        for epoch in range(1, epochs+1):
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
        results["per_task"][task] = {"accuracy": round(acc,4), "time_s": round(elapsed,1)}
        optimizer.zero_velocity()

    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs)/len(accs), 4)
    print("Mean accuracy: " + str(round(results["mean_accuracy"]*100, 2)))

    os.makedirs(results_dir, exist_ok=True)
    path = Path(results_dir) / f"fna_s{seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to " + str(path))
    return results


def train_lora(tasks, epochs, lora_rank, seed, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class LoRALayer(nn.Module):
        def __init__(self, linear, rank):
            super().__init__()
            self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
            self.bias   = nn.Parameter(linear.bias.data.clone(), requires_grad=False) if linear.bias is not None else None
            in_f, out_f = linear.in_features, linear.out_features
            self.A = nn.Parameter(torch.empty(rank, in_f))
            self.B = nn.Parameter(torch.zeros(out_f, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.scale = 1.0
        def forward(self, x):
            base = F.linear(x, self.weight, self.bias)
            lora = F.linear(F.linear(x, self.A), self.B) * self.scale
            return base + lora

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA only into ENCODER_QV (same 12 layers as FNA)
    for name, mod in model.named_modules():
        if name in ENCODER_QV and isinstance(mod, nn.Linear):
            parts  = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALayer(mod, lora_rank).to(device))

    model = model.to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    actual_count = sum(p.numel() for p in lora_params)
    print("Adapter: LoRA-manual | rank=" + str(lora_rank) + " | layers=" + str(len(ENCODER_QV)))
    print("Trainable params: " + str(actual_count) + " | lr=" + str(LORA_LR) + " | seed=" + str(seed))

    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    results = {
        "adapter": "lora", "tasks": tasks, "epochs": epochs,
        "rank": lora_rank, "lr": LORA_LR, "seed": seed,
        "trainable_params": actual_count, "per_task": {}
    }

    for task in tasks:
        print("\n--- Task: " + task + " ---")
        loader = make_loader(task, tokenizer, split="test")
        t0 = time.time()
        for epoch in range(1, epochs+1):
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
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(n,1):.4f}")
        acc     = evaluate(model, tokenizer, task, device)
        elapsed = time.time() - t0
        print("  Accuracy: " + str(round(acc*100,2)) + "%  (" + str(round(elapsed,1)) + "s)")
        results["per_task"][task] = {"accuracy": round(acc,4), "time_s": round(elapsed,1)}

    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs)/len(accs), 4)
    print("Mean accuracy: " + str(round(results["mean_accuracy"]*100, 2)))

    os.makedirs(results_dir, exist_ok=True)
    path = Path(results_dir) / f"lora_r{lora_rank}_s{seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to " + str(path))
    return results


def compare(results_dir, fna_seed, lora_rank, lora_seed):
    fna_path  = Path(results_dir) / f"fna_s{fna_seed}.json"
    lora_path = Path(results_dir) / f"lora_r{lora_rank}_s{lora_seed}.json"
    fna  = json.load(open(fna_path))
    lora = json.load(open(lora_path))
    print(f"\nFNA (196K params, nu={fna['nu']}) vs LoRA-r{lora_rank} ({lora['trainable_params']:,} params)")
    print(f"{'Task':<25} {'LoRA':>8} {'FNA':>8} {'Delta':>8}")
    print("-" * 55)
    for task in fna["per_task"]:
        f = fna["per_task"][task]["accuracy"]*100
        l = lora["per_task"][task]["accuracy"]*100
        print(f"  {task:<23} {l:>7.2f}% {f:>7.2f}% {f-l:>+7.2f}%")
    fm = fna["mean_accuracy"]*100
    lm = lora["mean_accuracy"]*100
    print("-" * 55)
    print(f"  {'MEAN':<23} {lm:>7.2f}% {fm:>7.2f}% {fm-lm:>+7.2f}%")
    print(f"\n  FNA params:  {fna['trainable_params']:,}")
    print(f"  LoRA params: {lora['trainable_params']:,}")



def format_superglue(example, task):
    if task == "boolq":
        text = "boolq question: " + example["question"] + " passage: " + example["passage"][:400]
        label = ["false", "true"][example["label"]]
    elif task == "cb":
        text = "cb premise: " + example["premise"] + " hypothesis: " + example["hypothesis"]
        label = ["entailment", "contradiction", "neutral"][example["label"]]
    elif task == "rte":
        text = "rte premise: " + example["premise"] + " hypothesis: " + example["hypothesis"]
        label = ["entailment", "not_entailment"][example["label"]]
    return text, label


def make_superglue_loader(task, tokenizer, split="validation", batch_size=16):
    from datasets import load_dataset
    ds = load_dataset("super_glue", task, split=split)
    inputs, targets = [], []
    for ex in ds:
        i, t = format_superglue(ex, task)
        inputs.append(i)
        targets.append(t)
    enc = tokenizer(inputs, truncation=True, max_length=MAX_INPUT,
                    padding="max_length", return_tensors="pt")
    lbl = tokenizer(targets, max_length=8,
                    padding="max_length", return_tensors="pt").input_ids
    lbl[lbl == tokenizer.pad_token_id] = -100
    dataset = torch.utils.data.TensorDataset(enc.input_ids, enc.attention_mask, lbl)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_superglue(model, tokenizer, task, device):
    from datasets import load_dataset
    ds = load_dataset("super_glue", task, split="validation")
    model.eval()

    if task == "boolq":
        label_words = ["false", "true"]
    elif task == "cb":
        label_words = ["entailment", "contradiction", "neutral"]
    elif task == "rte":
        label_words = ["entailment", "not_entailment"]

    label_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in label_words]

    correct, total = 0, 0
    loader = make_superglue_loader(task, tokenizer, split="validation", batch_size=32)

    for input_ids, attn_mask, labels in loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        outputs = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                 max_new_tokens=4, do_sample=False)
        for pred, label_row in zip(outputs, labels):
            pred_text = tokenizer.decode(pred, skip_special_tokens=True).strip().lower()
            gold = label_row[label_row != -100]
            if len(gold) == 0:
                continue
            gold_text = tokenizer.decode(gold, skip_special_tokens=True).strip().lower()
            if pred_text == gold_text:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def train_superglue(adapter, tasks, epochs, lora_rank, seed, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import math
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    if adapter == "fna":
        fna_layers = inject_fna_adapters(model, target_modules=ENCODER_QV,
                                         grid_size=GRID_SIZE, alpha=1.0, verbose=False)
        model = model.to(device)
        params = get_fna_optimizer_params(model)
        optimizer = FNAOptimizer(params, lr=FNA_LR, nu=FNA_NU)
        adapter_label = "fna"
    else:
        class LoRALayer(nn.Module):
            def __init__(self, linear, rank):
                super().__init__()
                self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
                self.bias   = nn.Parameter(linear.bias.data.clone(), requires_grad=False) if linear.bias is not None else None
                in_f, out_f = linear.in_features, linear.out_features
                self.A = nn.Parameter(torch.empty(rank, in_f))
                self.B = nn.Parameter(torch.zeros(out_f, rank))
                nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                self.scale = 1.0
            def forward(self, x):
                import torch.nn.functional as F
                base = F.linear(x, self.weight, self.bias)
                lora = F.linear(F.linear(x, self.A), self.B) * self.scale
                return base + lora

        for p in model.parameters():
            p.requires_grad = False
        for name, mod in model.named_modules():
            if name in ENCODER_QV and isinstance(mod, nn.Linear):
                parts  = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], LoRALayer(mod, lora_rank))

        model = model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=LORA_LR, weight_decay=0.01)
        adapter_label = "lora_r" + str(lora_rank)

    actual_count = sum(p.numel() for p in params)
    print("Adapter: " + adapter_label + " | params: " + str(actual_count) + " | seed=" + str(seed))

    results = {
        "adapter": adapter_label, "tasks": tasks, "epochs": epochs,
        "seed": seed, "trainable_params": actual_count, "per_task": {}
    }

    for task in tasks:
        print("\n--- SuperGLUE Task: " + task + " ---")
        loader = make_superglue_loader(task, tokenizer, split="train" if task != "boolq" else "train",
                                       batch_size=BATCH_SIZE)
        t0 = time.time()
        for epoch in range(1, epochs+1):
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
                if adapter == "lora":
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                total_loss += loss.item()
                n += 1
            print("  Epoch " + str(epoch) + "/" + str(epochs) + "  loss=" + str(round(total_loss/max(n,1), 4)))

        acc     = evaluate_superglue(model, tokenizer, task, device)
        elapsed = time.time() - t0
        print("  Accuracy: " + str(round(acc*100, 2)) + "%  (" + str(round(elapsed,1)) + "s)")
        results["per_task"][task] = {"accuracy": round(acc,4), "time_s": round(elapsed,1)}
        if adapter == "fna":
            optimizer.zero_velocity()

    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs)/len(accs), 4)
    print("Mean accuracy: " + str(round(results["mean_accuracy"]*100, 2)))

    os.makedirs(results_dir, exist_ok=True)
    path = Path(results_dir) / (adapter_label + "_superglue_s" + str(seed) + ".json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved to " + str(path))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",     choices=["fna","lora","compare","superglue_fna","superglue_lora"], default="fna")
    parser.add_argument("--tasks",       type=str, default="anatomy,astronomy,college_mathematics,high_school_physics")
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--lora_rank",   type=int, default=16)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--fna_seed",    type=int, default=42)
    parser.add_argument("--lora_seed",   type=int, default=42)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]

    if args.adapter == "fna":
        train_fna(tasks, args.epochs, args.lora_rank, args.seed, args.results_dir)
    elif args.adapter == "lora":
        train_lora(tasks, args.epochs, args.lora_rank, args.seed, args.results_dir)
    elif args.adapter == "compare":
        compare(args.results_dir, args.fna_seed, args.lora_rank, args.lora_seed)
    elif args.adapter == "superglue_fna":
        train_superglue("fna", tasks, args.epochs, args.lora_rank, args.seed, args.results_dir)
    elif args.adapter == "superglue_lora":
        train_superglue("lora", tasks, args.epochs, args.lora_rank, args.seed, args.results_dir)

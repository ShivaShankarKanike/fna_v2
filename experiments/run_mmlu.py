
import argparse, json, os, sys, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).parent.parent))
from fna import inject_fna_adapters, get_fna_optimizer_params, FNAOptimizer

MODEL_NAME  = "t5-small"
MAX_INPUT   = 512
MAX_TARGET  = 4
BATCH_SIZE  = 16
GRID_SIZE   = 128
ALPHA       = 1.0
LORA_RANK   = 8
FNA_LR      = 3e-4
FNA_NU      = 0.1515
LORA_LR     = 3e-4

# Only encoder q,v — 12 layers, fair comparison both sides
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

def format_mmlu(example):
    choices = example["choices"]
    labels  = ["A","B","C","D"]
    cs      = " ".join(f"{l}. {c}" for l,c in zip(labels,choices))
    return f"question: {example['question']} choices: {cs}", labels[example["answer"]]

def make_loader(task, tokenizer, split="test", batch_size=BATCH_SIZE):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", task, split=split, trust_remote_code=False)
    inputs, targets = [], []
    for ex in ds:
        i,t = format_mmlu(ex)
        inputs.append(i); targets.append(t)
    enc = tokenizer(inputs, truncation=True, max_length=MAX_INPUT,
                    padding="max_length", return_tensors="pt")
    lbl = tokenizer(targets, max_length=MAX_TARGET,
                    padding="max_length", return_tensors="pt").input_ids
    lbl[lbl == tokenizer.pad_token_id] = -100
    ds2 = torch.utils.data.TensorDataset(enc.input_ids, enc.attention_mask, lbl)
    return DataLoader(ds2, batch_size=batch_size, shuffle=False)

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
        outputs   = model.generate(input_ids=input_ids, attention_mask=attn_mask,
                                   max_new_tokens=1, do_sample=False)
        pred_ids  = outputs[:,1] if outputs.shape[1]>1 else outputs[:,0]
        for pred_id, label_row in zip(pred_ids, labels):
            gold = label_row[label_row != -100]
            if len(gold)==0: continue
            if pred_id.item()==gold[0].item(): correct+=1
            total+=1
    return correct/total if total>0 else 0.0

def train_fna(tasks, epochs, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Adapter: FNA | Device:", device)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    fna_layers = inject_fna_adapters(model, target_modules=ENCODER_QV,
                                     grid_size=GRID_SIZE, alpha=ALPHA, verbose=False)
    model = model.to(device)
    fna_params = get_fna_optimizer_params(model)
    optimizer  = FNAOptimizer(fna_params, lr=FNA_LR, nu=FNA_NU)
    total_trainable = sum(p.numel() for p in fna_params)
    print("Trainable params:", total_trainable)
    results = {"adapter":"fna","tasks":tasks,"grid_size":GRID_SIZE,
               "nu":FNA_NU,"trainable_params":total_trainable,"per_task":{}}
    for task in tasks:
        print("--- Task:", task, "---")
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
                total_loss += loss.item(); n+=1
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(n,1):.4f}")
        acc = evaluate(model, tokenizer, task, device)
        elapsed = time.time()-t0
        print(f"  Accuracy: {acc*100:.2f}%  ({elapsed:.1f}s)")
        results["per_task"][task] = {"accuracy":round(acc,4),"time_s":round(elapsed,1)}
        optimizer.zero_velocity()
    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs)/len(accs),4)
    os.makedirs(results_dir, exist_ok=True)
    with open(Path(results_dir)/"fna_results.json","w") as f:
        json.dump(results,f,indent=2)
    print("Mean accuracy:", results["mean_accuracy"]*100)

def train_lora(tasks, epochs, results_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from peft import get_peft_model, LoraConfig, TaskType
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Adapter: LoRA | Device:", device)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    # Target only encoder q,v to match FNA
    lora_cfg  = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=["q","v"],
        lora_dropout=0.0, bias="none",
        layers_to_transform=list(range(6)),  # encoder only (layers 0-5)
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)
    total_trainable = sum(p.numel() for p in lora_params)
    results = {"adapter":"lora","tasks":tasks,"rank":LORA_RANK,
               "lr":LORA_LR,"trainable_params":total_trainable,"per_task":{}}
    for task in tasks:
        print("--- Task:", task, "---")
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
                total_loss += loss.item(); n+=1
            print(f"  Epoch {epoch}/{epochs}  loss={total_loss/max(n,1):.4f}")
        acc = evaluate(model, tokenizer, task, device)
        elapsed = time.time()-t0
        print(f"  Accuracy: {acc*100:.2f}%  ({elapsed:.1f}s)")
        results["per_task"][task] = {"accuracy":round(acc,4),"time_s":round(elapsed,1)}
    accs = [v["accuracy"] for v in results["per_task"].values()]
    results["mean_accuracy"] = round(sum(accs)/len(accs),4)
    os.makedirs(results_dir, exist_ok=True)
    with open(Path(results_dir)/"lora_results.json","w") as f:
        json.dump(results,f,indent=2)
    print("Mean accuracy:", results["mean_accuracy"]*100)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", choices=["fna","lora"], default="fna")
    parser.add_argument("--tasks",   type=str, default="anatomy,astronomy")
    parser.add_argument("--epochs",  type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",")]
    if args.adapter=="fna": train_fna(tasks, args.epochs, args.results_dir)
    else: train_lora(tasks, args.epochs, args.results_dir)

"""Sweep all sparsities across 10 scenarios — PPL + GPU memory + latency.

Run in SliceGPT venv:
    cd baselines/SliceGPT
    uv run python rap_baseline_scripts/sweep_scenarios.py
"""
import gc
import json
import math
import sys
import time
from pathlib import Path

import pynvml
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, "src")
from slicegpt import hf_utils

HF_MODEL = "meta-llama/Llama-2-7b-hf"
SPARSITIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
SCENARIOS = [
    {"id": 1,  "bs": 1,  "seq_len": 4096},
    {"id": 2,  "bs": 2,  "seq_len": 2048},
    {"id": 3,  "bs": 4,  "seq_len": 2048},
    {"id": 4,  "bs": 4,  "seq_len": 4096},
    {"id": 5,  "bs": 8,  "seq_len": 1024},
    {"id": 6,  "bs": 8,  "seq_len": 2048},
    {"id": 7,  "bs": 8,  "seq_len": 4096},
    {"id": 8,  "bs": 16, "seq_len": 512},
    {"id": 9,  "bs": 16, "seq_len": 1024},
    {"id": 10, "bs": 16, "seq_len": 2048},
]


def get_gpu_memory_gb(idx=0):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlShutdown()
    return m.used / 1024**3


def load_eval_dataloader(tokenizer, dataset_name, seq_len=2048, batch_size=2):
    if dataset_name == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [r["text"] for r in ds if r["text"].strip()]
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", split="test", revision="refs/convert/parquet")
        texts = [r["sentence"] for r in ds if r["sentence"].strip()]
    full_text = " ".join(texts)
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)
    ids = tokens["input_ids"].squeeze(0)
    n = (len(ids) // seq_len) * seq_len
    ids = ids[:n].reshape(-1, seq_len)
    return DataLoader(TensorDataset(ids, torch.ones_like(ids)), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def eval_ppl(model, dataloader, device):
    model.eval()
    total_loss, total_tokens, n_samples = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    wb = next(iter(dataloader))
    model(input_ids=wb[0].to(device), attention_mask=wb[1].to(device))
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for batch in dataloader:
        ids = batch[0].to(device)
        mask = batch[1].to(device)
        logits = model(input_ids=ids, attention_mask=mask).logits
        sl = logits[:, :-1, :].contiguous()
        labels = ids[:, 1:].contiguous()
        loss = loss_fn(sl.view(-1, sl.size(-1)), labels.view(-1))
        sm = mask[:, 1:].contiguous().view(-1)
        total_loss += (loss * sm).sum().item()
        total_tokens += sm.sum().item()
        n_samples += ids.size(0)
    end.record()
    torch.cuda.synchronize()
    lat = start.elapsed_time(end) / 1000.0
    ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return ppl, lat, n_samples


def measure_mem(sparsity, bs, seq_len):
    """Spawn clean subprocess to measure GPU memory."""
    import subprocess
    cmd = [
        sys.executable, "rap_baseline_scripts/measure_memory_worker.py",
        "--sliced-model-path", f"sliced_models/llama2_{sparsity}",
        "--sparsity", str(sparsity),
        "--bs", str(bs), "--seq-len", str(seq_len),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("{"):
            data = json.loads(line)
            if data["status"] == "OOM":
                return "OOM"
            return data["mem_peak_gb"]
    return "ERROR"


def main():
    torch.manual_seed(42)
    out_path = Path("../../outputs/dynamic_memory_comparison_v2/slicegpt.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for sp in SPARSITIES:
        print(f"\n  sparsity={sp}", flush=True)
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            HF_MODEL, f"sliced_models/llama2_{sp}", sparsity=sp, round_interval=8,
        )
        model = model_adapter.model.to("cuda")
        model.eval()

        ppl_results, latency_results = {}, {}
        for ds in ["wikitext2", "ptb"]:
            dl = load_eval_dataloader(tokenizer, ds)
            ppl, lat, ns = eval_ppl(model, dl, torch.device("cuda"))
            ppl_results[ds] = round(ppl, 2)
            latency_results[ds] = {"latency_sec": round(lat, 2), "n_samples": ns}
            print(f"    {ds}: PPL={ppl:.2f}, lat={lat:.2f}s", flush=True)

        mem_at = {}
        for s in SCENARIOS:
            m = measure_mem(sp, s["bs"], s["seq_len"])
            mem_at[s["id"]] = round(m, 2) if isinstance(m, (int, float)) else m
            print(f"    S{s['id']} (bs={s['bs']},sl={s['seq_len']}): {mem_at[s['id']]}GB", flush=True)

        results.append({"sparsity": sp, "ppl": ppl_results, "latency": latency_results, "memory_at_scenarios": mem_at})
        del model, model_adapter, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

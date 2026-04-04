"""Evaluate SliceGPT sliced model — PPL and GPU memory (nvidia-smi).

Uses the same evaluation logic as RAP for fair comparison:
- WikiText-2 test split, concatenated and chunked to seq_len
- Global token-level average NLL → exp → PPL
- GPU memory via nvidia-smi (process-level)

Usage:
    cd baselines/SliceGPT
    uv run python rap_baseline_scripts/eval_ppl.py --sliced-model-path sliced_models/llama2_0.4 --sparsity 0.4
    uv run python rap_baseline_scripts/eval_ppl.py  # dense model
"""
import argparse
import gc
import math
import subprocess
import sys
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from slicegpt import hf_utils


def load_eval_dataloader(tokenizer, dataset_name="wikitext2", seq_len=2048, batch_size=8):
    """Load test split — same logic as RAP's load_calibration_dataset."""
    if dataset_name == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        texts = [row["text"] for row in ds if row["text"].strip()]
    elif dataset_name == "ptb":
        ds = load_dataset("ptb_text_only", split="test", revision="refs/convert/parquet")
        texts = [row["sentence"] for row in ds if row["sentence"].strip()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext2' or 'ptb'.")

    full_text = " ".join(texts)
    tokens = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = tokens["input_ids"].squeeze(0)

    n_tokens = (len(input_ids) // seq_len) * seq_len
    input_ids = input_ids[:n_tokens].reshape(-1, seq_len)
    attention_mask = torch.ones_like(input_ids)

    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device):
    """Evaluate PPL — same logic as RAP's evaluate_perplexity."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    for batch in dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        shift_mask = attention_mask[:, 1:].contiguous().view(-1)
        loss = loss * shift_mask
        total_tokens += shift_mask.sum().item()
        total_loss += loss.sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def get_nvidia_smi_memory():
    """Get GPU memory usage from nvidia-smi in GB."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return int(result.stdout.strip()) / 1024
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate SliceGPT model (RAP-consistent)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--sliced-model-path", type=str, default=None)
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--round-interval", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    datasets_to_eval = ["wikitext2", "ptb"]

    # Load model
    print(f"Model: {args.model}", flush=True)
    if args.sliced_model_path:
        print(f"Sliced model: {args.sliced_model_path}, sparsity={args.sparsity}", flush=True)
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model, args.sliced_model_path,
            sparsity=args.sparsity, round_interval=args.round_interval,
        )
        model = model_adapter.model
    else:
        print("Dense model (no slicing)", flush=True)
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, dtype=torch.float16)
        model = model_adapter.model

    model.to(args.device)
    model.eval()

    # Evaluate on all datasets
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    ppl_results = {}
    total_time = 0.0
    for ds_name in datasets_to_eval:
        dataloader = load_eval_dataloader(tokenizer, dataset_name=ds_name, seq_len=args.seq_len, batch_size=args.batch_size)
        t0 = time.time()
        ppl = evaluate_perplexity(model, dataloader, torch.device(args.device))
        elapsed = time.time() - t0
        ppl_results[ds_name] = ppl
        total_time += elapsed
        print(f"  {ds_name}: PPL={ppl:.2f} ({elapsed:.1f}s)", flush=True)

    torch.cuda.synchronize()
    gpu_nvidia_smi = get_nvidia_smi_memory()

    print(f"\n{'='*40}", flush=True)
    for ds_name, ppl in ppl_results.items():
        print(f"  PPL ({ds_name}): {ppl:.2f}", flush=True)
    print(f"  GPU (nvidia-smi): {gpu_nvidia_smi:.2f} GB", flush=True)
    print(f"  Total eval time: {total_time:.1f} s", flush=True)
    print(f"{'='*40}", flush=True)


if __name__ == "__main__":
    main()

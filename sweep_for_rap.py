"""Sweep SliceGPT at multiple sparsity levels for RAP comparison.

Run in the SliceGPT venv:
    uv run python sweep_for_rap.py

Outputs: ../../outputs/dynamic_memory_comparison/slicegpt_sweep.json
"""
import gc
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "src")

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

MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def measure_peak_memory(model, bs, seq_len, device="cuda"):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    input_ids = torch.randint(1, 32000, (bs, seq_len), device=device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except torch.cuda.OutOfMemoryError:
            del input_ids
            torch.cuda.empty_cache()
            return "OOM"

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del input_ids
    torch.cuda.empty_cache()
    return peak


def evaluate_ppl_wikitext2(model, tokenizer, device, max_seq_len=2048):
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"].squeeze(0)

    n_tokens = (len(input_ids) // max_seq_len) * max_seq_len
    input_ids = input_ids[:n_tokens].reshape(-1, max_seq_len)

    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), 2):
            batch = input_ids[i:i+2].to(device)
            outputs = model(input_ids=batch)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.sum().item()
            total_tokens += loss.numel()

    return math.exp(total_loss / total_tokens)


def slice_model(sparsity):
    """Slice LLaMA-2 7B at given sparsity using SliceGPT. Caches to disk."""
    from slicegpt import data_utils, hf_utils, layernorm_fusion, rotate
    from slicegpt.rotate import ConstSlicingScheduler

    save_dir = Path(f"sliced_models/llama2_{sparsity}")
    ckpt_path = save_dir / "model.pt"

    # Load from cache if available
    if ckpt_path.exists():
        print(f"  Loading cached sliced model from {ckpt_path}", flush=True)
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(MODEL_NAME, dtype=torch.float16)
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)
        new_dim = int((1 - sparsity) * model_adapter.hidden_size)
        new_dim -= new_dim % 8
        scheduler = ConstSlicingScheduler(new_dim)
        rotate.slice_rotated_model(model_adapter, scheduler)
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model_adapter.model.load_state_dict(state_dict)
        return model_adapter.model, tokenizer

    print(f"  Slicing with sparsity={sparsity}...", flush=True)
    model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(MODEL_NAME, dtype=torch.float16)

    # Calibration data
    dataset = data_utils.get_dataset("wikitext2")
    train_loader = data_utils.prepare_dataloader(
        dataset=dataset["train"], tokenizer=tokenizer,
        max_seqlen=2048, batch_size=16, nsamples=128, seed=42,
    )

    # Replace layers + fuse layernorms
    layernorm_fusion.replace_layers(model_adapter)
    layernorm_fusion.fuse_modules(model_adapter)

    # Compute new embedding dimension
    new_dim = int((1 - sparsity) * model_adapter.hidden_size)
    new_dim -= new_dim % 8
    print(f"  New embedding dimension: {new_dim} (actual sparsity: {100*(1 - new_dim / model_adapter.hidden_size):.2f}%)", flush=True)

    # Rotate and slice
    scheduler = ConstSlicingScheduler(new_dim)
    rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="random")

    # Save state dict
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_adapter.model.state_dict(), str(ckpt_path))
    print(f"  Saved sliced model to {ckpt_path} ({ckpt_path.stat().st_size / 1e9:.1f}GB)", flush=True)

    return model_adapter.model, tokenizer


def main():
    torch.manual_seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path("../../outputs/dynamic_memory_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slicegpt_sweep.json"

    # Resume from existing results
    results = []
    done_sparsities = set()
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        for r in existing:
            if r.get("status") == "ok":
                results.append(r)
                done_sparsities.add(r["sparsity"])
        print(f"Resuming: {len(done_sparsities)} sparsities already done", flush=True)

    for sparsity in SPARSITIES:
        if sparsity in done_sparsities:
            print(f"\n  Skipping sparsity={sparsity} (already done)", flush=True)
            continue
        print(f"\n{'='*50}", flush=True)
        print(f"  SliceGPT sparsity={sparsity}", flush=True)
        print(f"{'='*50}", flush=True)

        try:
            model, tokenizer = slice_model(sparsity)
        except Exception as e:
            print(f"  Slicing failed: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"sparsity": sparsity, "status": "failed", "error": str(e)[:200]})
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            continue

        model.half().cuda()
        model.eval()
        model.config.use_cache = False

        # PPL
        try:
            ppl = evaluate_ppl_wikitext2(model, tokenizer, "cuda")
            print(f"  PPL (WikiText-2): {ppl:.2f}", flush=True)
        except Exception as e:
            print(f"  PPL eval failed: {e}", flush=True)
            ppl = None

        # Memory at each scenario
        memory_at_scenarios = {}
        for s in SCENARIOS:
            peak = measure_peak_memory(model, s["bs"], s["seq_len"])
            if peak == "OOM":
                memory_at_scenarios[s["id"]] = "OOM"
                print(f"    Scenario {s['id']} (bs={s['bs']},sl={s['seq_len']}): OOM", flush=True)
            else:
                memory_at_scenarios[s["id"]] = round(peak / 1e9, 2)
                print(f"    Scenario {s['id']} (bs={s['bs']},sl={s['seq_len']}): {peak/1e9:.2f}GB", flush=True)

        results.append({
            "sparsity": sparsity, "status": "ok",
            "ppl_wikitext2": round(ppl, 2) if ppl else None,
            "memory_at_scenarios": memory_at_scenarios,
        })

        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone! Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

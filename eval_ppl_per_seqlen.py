"""Re-evaluate SliceGPT PPL at different seq_lens for fair comparison.

Usage: uv run python eval_ppl_per_seqlen.py
"""
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, "src")

SPARSITIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
SEQ_LENS = [512, 1024, 2048, 4096]
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def evaluate_ppl(model, tokenizer, device, seq_len):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"].squeeze(0)

    n_tokens = (len(input_ids) // seq_len) * seq_len
    input_ids = input_ids[:n_tokens].reshape(-1, seq_len)

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


def main():
    torch.manual_seed(42)
    from slicegpt import data_utils, hf_utils, layernorm_fusion, rotate
    from slicegpt.rotate import ConstSlicingScheduler

    out_path = Path("../../outputs/dynamic_memory_comparison/slicegpt_ppl_per_seqlen.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out_path.exists():
        results = json.loads(out_path.read_text())

    for sparsity in SPARSITIES:
        sp_key = str(sparsity)
        if sp_key in results and len(results[sp_key]) == len(SEQ_LENS):
            print(f"Skipping sparsity={sparsity} (already done)", flush=True)
            continue

        print(f"\n{'='*50}", flush=True)
        print(f"  SliceGPT sparsity={sparsity}", flush=True)
        print(f"{'='*50}", flush=True)

        # Slice model
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(MODEL_NAME, dtype=torch.float16)
        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)
        new_dim = int((1 - sparsity) * model_adapter.hidden_size)
        new_dim -= new_dim % 8
        scheduler = ConstSlicingScheduler(new_dim)

        ckpt_path = Path(f"sliced_models/llama2_{sparsity}/model.pt")
        if ckpt_path.exists():
            print(f"  Loading cached model from {ckpt_path}", flush=True)
            rotate.slice_rotated_model(model_adapter, scheduler)
            state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
            model_adapter.model.load_state_dict(state_dict)
        else:
            dataset = data_utils.get_dataset("wikitext2")
            train_loader = data_utils.prepare_dataloader(
                dataset=dataset["train"], tokenizer=tokenizer,
                max_seqlen=2048, batch_size=16, nsamples=128, seed=42,
            )
            rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="random")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_adapter.model.state_dict(), str(ckpt_path))
            print(f"  Saved to {ckpt_path}", flush=True)

        model = model_adapter.model
        model.half().cuda()
        model.eval()
        model.config.use_cache = False

        sp_results = {}
        for sl in SEQ_LENS:
            ppl = evaluate_ppl(model, tokenizer, "cuda", sl)
            sp_results[str(sl)] = round(ppl, 2)
            print(f"  seq_len={sl}: PPL={ppl:.2f}", flush=True)

        results[sp_key] = sp_results
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        del model, model_adapter, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone! Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

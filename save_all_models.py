"""Slice and save models at all sparsity levels."""
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, "src")

SPARSITIES = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def main():
    torch.manual_seed(42)
    from slicegpt import data_utils, hf_utils, layernorm_fusion, rotate
    from slicegpt.rotate import ConstSlicingScheduler

    for sparsity in SPARSITIES:
        ckpt_path = Path(f"sliced_models/llama2_{sparsity}/model.pt")
        if ckpt_path.exists():
            size_gb = ckpt_path.stat().st_size / 1e9
            print(f"Skipping sparsity={sparsity} (cached, {size_gb:.1f}GB)", flush=True)
            continue

        print(f"\nSlicing sparsity={sparsity}...", flush=True)
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(MODEL_NAME, dtype=torch.float16)

        dataset = data_utils.get_dataset("wikitext2")
        train_loader = data_utils.prepare_dataloader(
            dataset=dataset["train"], tokenizer=tokenizer,
            max_seqlen=2048, batch_size=16, nsamples=128, seed=42,
        )

        layernorm_fusion.replace_layers(model_adapter)
        layernorm_fusion.fuse_modules(model_adapter)

        new_dim = int((1 - sparsity) * model_adapter.hidden_size)
        new_dim -= new_dim % 8
        scheduler = ConstSlicingScheduler(new_dim)
        rotate.rotate_and_slice(model_adapter, train_loader, scheduler, final_orientation="random")

        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model_adapter.model.state_dict(), str(ckpt_path))
        size_gb = ckpt_path.stat().st_size / 1e9
        print(f"  Saved {ckpt_path} ({size_gb:.1f}GB)", flush=True)

        del model_adapter, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()

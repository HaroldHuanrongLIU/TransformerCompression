"""Worker: load sliced model, forward at (bs, seq_len), print peak memory."""
import argparse
import gc
import json
import sys

import pynvml
import torch

sys.path.insert(0, "src")
from slicegpt import hf_utils

HF_MODEL = "meta-llama/Llama-2-7b-hf"


def get_gpu_memory_gb(idx=0):
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(idx)
    m = pynvml.nvmlDeviceGetMemoryInfo(h)
    pynvml.nvmlShutdown()
    return m.used / 1024**3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sliced-model-path", type=str, required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    args = parser.parse_args()

    model_adapter, _ = hf_utils.load_sliced_model(
        HF_MODEL, args.sliced_model_path, sparsity=args.sparsity, round_interval=8,
    )
    model = model_adapter.model.to("cuda")
    model.eval()

    torch.cuda.empty_cache()
    gc.collect()

    mem_before = get_gpu_memory_gb()
    input_ids = torch.randint(1, 32000, (args.bs, args.seq_len), device="cuda")
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
            torch.cuda.synchronize()
            mem_peak = get_gpu_memory_gb()
            print(json.dumps({"status": "ok", "mem_before_gb": round(mem_before, 2), "mem_peak_gb": round(mem_peak, 2)}))
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"status": "OOM"}))


if __name__ == "__main__":
    main()

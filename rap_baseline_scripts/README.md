# RAP Baseline Scripts — SliceGPT

Scripts for evaluating SliceGPT as a baseline for the RAP project. All scripts use **identical evaluation logic** as RAP to ensure fair comparison.

## Scripts

### eval_ppl.py

Evaluate perplexity and GPU memory of a sliced (or dense) model.

**Evaluation protocol** (same as RAP):
- Dataset: WikiText-2 or PTB test split
- Tokenization: concatenate all text, chunk into `seq_len` sequences
- Loss: CrossEntropyLoss per token, masked by attention_mask
- PPL: exp(mean of all token-level losses)
- GPU memory: nvidia-smi process-level measurement

**Usage:**

```bash
cd baselines/SliceGPT

# Dense model (wikitext2)
uv run python rap_baseline_scripts/eval_ppl.py

# Sliced model (wikitext2)
uv run python rap_baseline_scripts/eval_ppl.py \
  --sliced-model-path sliced_models/llama2_0.4 \
  --sparsity 0.4

# PTB dataset
uv run python rap_baseline_scripts/eval_ppl.py \
  --sliced-model-path sliced_models/llama2_0.4 \
  --sparsity 0.4 \
  --dataset ptb

# Custom batch size and seq length
uv run python rap_baseline_scripts/eval_ppl.py \
  --sliced-model-path sliced_models/llama2_0.4 \
  --sparsity 0.4 \
  --batch-size 4 \
  --seq-len 1024
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | meta-llama/Llama-2-7b-hf | Base HuggingFace model |
| `--sliced-model-path` | None | Path to sliced model dir (None = dense) |
| `--sparsity` | 0.0 | Sparsity level used for slicing |
| `--round-interval` | 8 | Rounding interval for embedding dimension |
| `--batch-size` | 8 | Batch size for evaluation |
| `--seq-len` | 2048 | Sequence length |
| `--dataset` | wikitext2 | Evaluation dataset: `wikitext2` or `ptb` |
| `--device` | cuda | PyTorch device |

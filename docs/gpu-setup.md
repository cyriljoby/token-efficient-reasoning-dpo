# GPU run (RunPod)

Everything up to pair construction runs on CPU. This is only for rollout
generation at scale, training, and evaluation.

Pod: **RTX 4090**, PyTorch/CUDA template, ~50GB volume. Billed by the second --
stop it when idle.

## Setup

```bash
git clone https://github.com/cyriljoby/token-efficient-reasoning-dpo.git
cd token-efficient-reasoning-dpo

# Torch comes with the template and is built for its CUDA version.
# Do NOT `pip install -r requirements.txt` -- that file pins the macOS build.
pip install vllm trl datasets accelerate peft

python -c "import torch, vllm; print(torch.__version__, torch.cuda.get_device_name(0))"
```

Versions from the run (RunPod PyTorch template, RTX 4090):

```
torch        2.13.0+cu130
vllm         0.28.0
trl          1.12.0
transformers 5.16.1
datasets     5.0.1
gpu          NVIDIA GeForce RTX 4090
```

Note trl, transformers and datasets match the local versions in
`requirements.txt`, so the custom DPO objective is validated against the same
TRL the pod trains with.

## 1. Parity check (~5 min)

The vLLM backend has never executed -- it was written without a CUDA machine to
test on. Run the pilot config first and compare against the known-good local
result before spending anything on a full run.

```bash
python -m data.generate_rollouts \
  --backend vllm --n-prompts 15 --n-rollouts 6 \
  --out data/rollouts/parity.jsonl
```

Expected, from the local pilot (n=15 prompts, 90 rollouts):

| quantity | local pilot |
|---|---|
| pass rate | 4-6 of 6 on most prompts |
| truncation | ~13% |
| `no_answer` | 12/90 |
| grade method | mostly `boxed`, no parse failures |
| median length ratio | ~1.35x |
| mean tokens (correct, untruncated) | ~496 |

Sampling is stochastic, so these will not match exactly. Large divergence --
especially near-zero truncation or a collapsed length spread -- means the vLLM
path differs from the HF path and should be investigated before scaling.

## 2. Full generation

```bash
tmux new -s rollouts
python -m data.generate_rollouts \
  --backend vllm --n-prompts 2000 --n-rollouts 6 \
  --max-new-tokens 2048 --temperature 0.8 \
  --out data/rollouts/train.jsonl --resume
```

`--resume` skips prompt indices already in the output file, so a disconnect or
a reclaimed pod costs a chunk, not the run. Safe to re-run the same command.

The fixed-seed shuffle means raising `--n-prompts` later extends the same
prefix rather than reshuffling, so scaling up also resumes cleanly.

## 3. Pairs (CPU, seconds)

```bash
python -m data.build_pairs --in data/rollouts/train.jsonl \
  --out data/pairs/train.jsonl
```

Also build the filtered variant to compare pair quality against quantity:

```bash
python -m data.build_pairs --in data/rollouts/train.jsonl \
  --min-ratio 1.5 --out data/pairs/train_ratio15.jsonl
```

## 4. Getting results off the pod

The volume is not backup. Rollouts are the expensive artifact -- everything
downstream is cheap to recompute from them.

```bash
# data/rollouts/ and data/pairs/ are gitignored, so copy them out directly
runpodctl send data/rollouts/train.jsonl
```

## Notes

- `--chunk-size` trades batching efficiency against checkpoint frequency.
- Stop the pod when idle.

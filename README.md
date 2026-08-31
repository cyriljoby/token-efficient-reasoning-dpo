# Preference Optimization for Token-Efficient LLM Reasoning

Baselines:
- Base Model
- SFT-on-chosen
- Concise-prompt

Evaluating different methods for token-efficiency:
- DPO
    - First using TRL and then benchmarking against scratch-written dpo
- SimPO (length normalized DPO without a reference tether)
- Strech Goal: GRPO (true online RL)

## Results

```
TBD — not yet run.

```

## What I built

TBD.

## What I found

TBD.

## Repo layout

```
data/        preference pair generation, splits
training/    TRL pipeline, custom DPO objective
docs/        background notes
```

# Preference Optimization for Token-Efficient LLM Reasoning

Does DPO reduce reasoning-token generation in a small open-weight reasoning
model, and does it beat simply asking the model to be concise?

Short answer: it reduces tokens, and no, it does not beat the prompt.

**Model** DeepSeek-R1-Distill-Qwen-1.5B · **Data** GSM8K · **Hardware** one RTX 4090

## Results

All conditions: greedy decoding, `max_new_tokens=2048`, the full 1,319-problem
GSM8K test split, generated through one code path. Intervals are percentile
bootstrap over problems.

| condition | tokens | vs base | accuracy | reasoning | truncated |
|---|---:|---:|---|---:|---:|
| base | 675 | — | 0.757 [0.734, 0.779] | 436 | 8.2% |
| DPO lr 5e-6 | 612 | −9.3% | 0.770 [0.748, 0.793] | 365 | 6.0% |
| DPO lr 1e-5 | 512 | −24.2% | 0.751 [0.728, 0.774] | 257 | 4.0% |
| DPO lr 2e-5 | 352 | −47.9% | 0.727 [0.704, 0.751] | 128 | 0.4% |
| DPO lr 5e-5 | 167 | −75.2% | 0.629 [0.603, 0.654] | 56 | 0.0% |
| concise prompt A | 365 | −45.9% | 0.742 [0.719, 0.766] | 137 | 0.1% |
| concise prompt B | 408 | −39.6% | 0.745 [0.722, 0.769] | 154 | 0.2% |
| concise prompt C | 414 | −38.7% | 0.752 [0.729, 0.776] | 151 | 0.1% |

Conditions share the same 1,319 problems, so accuracy costs are tested paired
(10,000 bootstrap resamples of per-problem differences). Positive = worse than
base:

| condition | accuracy cost | | tokens saved |
|---|---|---|---:|
| DPO lr 5e-6 | −0.0136 [−0.0288, +0.0015] | n.s. | 63 [45, 81] |
| DPO lr 1e-5 | +0.0053 [−0.0136, +0.0243] | n.s. | 163 [141, 186] |
| concise prompt C | +0.0045 [−0.0167, +0.0258] | n.s. | 261 [236, 287] |
| concise prompt A | +0.0144 [−0.0076, +0.0371] | n.s. | 310 [285, 335] |
| DPO lr 2e-5 | +0.0296 [+0.0061, +0.0523] | **significant** | 324 [299, 350] |
| DPO lr 5e-5 | +0.1274 [+0.1016, +0.1531] | **significant** | 508 [482, 535] |

The unpaired view puts DPO at 2e-5 (−47.9%) alongside the best prompt (−45.9%)
with overlapping intervals. Pairing removes that: 2e-5 costs a real 3 accuracy
points, and it was the only DPO setting whose savings rivalled prompting.

## What I found

**1. DPO reduces tokens, monotonically with learning rate.** −9.3% → −24.2% →
−47.9% → −75.2% across 5e-6 to 5e-5. Well-behaved, not a lottery.

**2. Accuracy slides, then falls off a cliff.** 0.770 → 0.751 → 0.727 is within
noise of base; 0.727 → 0.629 between 2e-5 and 5e-5 is a collapse.

**3. A one-line prompt beats it.** At zero measurable accuracy cost, prompting
saves 310 tokens against DPO's 163 — roughly double, with no training, no data
generation, and no GPU. Prompt C strictly dominates DPO at 1e-5: fewer tokens
(414 vs 512) *and* equal accuracy (0.752 vs 0.751).

**4. Training metrics can look perfect while the model degrades.** The 5e-5 run
reached 100% training preference accuracy — and lost 12.7 accuracy points at
eval. Its policy drifted 146 nats from the reference and token accuracy fell
0.937 → 0.824 during training. Reasoning collapsed 436 → 56 tokens while the
output format stayed intact: the model learned to skip thinking, not to break.

**5. The base model's long outputs are second-guessing spirals, not long
solutions.** Truncated completions contain "Wait" 13× on average versus 0.5× for
completions that finish, concentrated in ambiguous problems. Median clean
completion uses 18% of the token budget. Every intervention reduces this;
prompting nearly eliminates it (8.2% → 0.1%).

**6. The reduction is in reasoning, not answers.** At 1e-5, reasoning tokens fall
41% (436 → 257) while answer tokens rise slightly (238 → 254). Answers are ~40%
of all output, so total reduction understates the effect on reasoning.

## What I built

**Custom DPO objective** (`training/dpo_loss.py`) — masked sequence log-probs
under policy and frozen reference, with sum and mean (length-normalized)
variants. Validated against TRL to 1e-5 absolute on real batches from a live
`DPOTrainer`; measured agreement is 1e-6 relative, at floating-point round-off.
The log-prob computation is checked separately from the final scalar, since
masking and off-by-one errors can cancel in the scalar.

**Evaluation layer** (`evaluation/`) — answer extraction with LaTeX, currency and
separator normalization, preferring `\boxed{}` and refusing ambiguous parses
rather than guessing; reasoning/answer token split at `</think>`; truncation and
never-answered rates per condition.

**Data pipeline** (`data/`) — 12,000 rollouts (2,000 prompts × 6, temperature
0.8) generated in 10.2 minutes via vLLM, scored, and paired shortest-correct
against longest-correct. 1,809 pairs, 90.5% prompt yield, median length ratio
1.62×. Generation and pairing are separate passes so pairs can be rebuilt under
different rules without regenerating tokens.

**Training** (`training/`) — LoRA (r=16, 18.5M of 1.8B parameters trainable) DPO
via TRL, with the adapter-disabled base serving as the reference model.

## What the process caught

- **The grader scored unfinished reasoning.** The chat template injects `<think>`
  into the *prompt*, so completions carry no opening tag — which made a branch
  unreachable and sent truncated generations down a path that graded them on
  whatever number appeared last mid-thought. 4 false positives per 90 rollouts.
- **`pad_token_id == eos_token_id`** (both 151643). Stripping padding also
  stripped the real EOS, which would have marked every completion truncated.
- **The R1 chat template drops `<think>…</think>` when rendering an assistant
  turn.** Conversational-format training data would have had every reasoning
  trace silently deleted.
- **An OOM in the 152k-vocabulary logits tensor**, not in model weights — LoRA
  had already reduced parameter memory to ~150MB; the constraint was activations.

## Reproducing

```bash
pip install -r requirements.txt          # torch pinned for macOS; see docs/gpu-setup.md for CUDA
python -m data.generate_rollouts --backend vllm --n-prompts 2000 --resume \
  --out data/rollouts/train.jsonl
python -m data.build_pairs --in data/rollouts/train.jsonl --out data/pairs/train.jsonl
python -m training.train_dpo --pairs data/pairs/train.jsonl --output-dir outputs/dpo \
  --batch-size 1 --grad-accum 16 --learning-rate 1e-5
python -m training.merge_adapter --adapter outputs/dpo --out models/dpo
python -m evaluation.run_eval --model models/dpo --condition dpo --out data/eval/dpo.jsonl
python -m evaluation.summarize data/eval/*.jsonl
```

Tests: `pytest tests/ -q` (77 tests, CPU only, no GPU required).

## Limitations

- **One training seed per learning rate.** The shape of the curve across four
  points is robust; individual accuracy differences of a point or two are not.
- **No SFT-on-chosen control.** Strictly, this shows DPO reduces tokens, not that
  DPO's preference machinery is responsible rather than fine-tuning on the chosen
  responses generally.
- **No pairing-cancellation ablation.** Pairs are shortest-correct vs
  longest-correct only, so mixed correct/incorrect pairs are untested.
- **Bootstrap intervals quantify problem sampling**, not training variance.
- **GSM8K only**, one model, one prompt format.

## Prior work

This is a reproduction and empirical characterization, not a novel technique.

- Rafailov et al. 2023, *Direct Preference Optimization* — the objective used here.
- Park et al. 2024, *Disentangling Length from Quality in DPO* — DPO's documented
  length bias toward longer outputs, which runs opposite to this project's goal.
- Meng et al. 2024, *SimPO* — length-normalized preference optimization.
- Hong et al. 2025, *Pruning Long Chain-of-Thought of Large Reasoning Models via
  Small-Scale Preference Optimization* (arXiv:2508.10164) — proposes LCPO for
  this task, reporting >50% length reduction. Read at abstract level only.

## Repo layout

```
data/        rollout generation, preference pair construction
training/    custom DPO objective, LoRA training, adapter merging
evaluation/  grading, token counting, eval runner, aggregation
tests/       objective validation against TRL, grader and counter fixtures
docs/        DPO and LoRA notes, GPU setup
```

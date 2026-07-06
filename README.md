# Chain-of-Thought's Double Edge Is Model-Specific: Prompting versus Scale Across Three Model Families

A controlled study of when chain-of-thought (CoT) prompting helps and when it *hurts*, and how that
depends on the model family. Three instruction-tuned families are swept across sizes and four
prompting strategies, with 4-bit quantization held constant, run locally via MLX on a single laptop.

**Author:** Samir Samal (Independent Researcher)

> **Status:** Complete. This repository matches the arXiv preprint: three families evaluated on a
> **single fixed evaluation set held constant across every model and strategy** (the full SST-2
> validation set, 872 items; a fixed 1000-item sample each for GSM8K and MMLU), under greedy
> decoding, with 95% Wilson confidence intervals and Holm-adjusted McNemar tests. See
> [`paper.pdf`](paper.pdf) for the write-up.
>
> An earlier two-seed variant (pooled n = 600 per condition) is preserved on the
> **`n600-two-seed-backup`** branch.

## Question

When a single prompting technique (chain-of-thought) helps one model on a task, is it safe to assume
it helps another? We sweep three instruction-tuned model families across sizes and four prompting
strategies (zero-shot, few-shot, CoT, structured/JSON) on three task types: SST-2 (sentiment),
MMLU (knowledge), and GSM8K (math reasoning). The study also quantifies a **prompt–parameter
exchange rate**: how much model scaling a prompting strategy substitutes for.

## Families

| Family | Sizes |
|---|---|
| Qwen2.5-Instruct | 0.5B / 1.5B / 3B / 7B |
| Llama-3 | 1B / 3B / 8B |
| Gemma-2 | 2B / 9B |

## Findings

- **Primary: chain-of-thought's effect on knowledge is model-specific, and three families show three
  distinct patterns.** Qwen2.5: CoT *degrades* MMLU increasingly with scale (−7.2 to −15.5 pp).
  Gemma-2: CoT harms the small model but the harm *fades* with scale (−16.1 pp at 2B, +0.6 pp at 9B).
  Llama-3: CoT stays small with no scale trend (+0.9, +3.0, −5.2 pp). Not only *whether* CoT hurts
  knowledge, but *how that changes with scale*, is family-dependent. Pooling these unlike responses
  into one average, as a broad meta-analysis does, describes none of them.
- **Confirming prior work** (Sprague et al., 2024): prompting substitutes for scale on reasoning but
  barely on knowledge. On GSM8K, CoT at a small size beats zero-shot at a much larger size
  (Qwen-1.5B CoT 58.7% vs Qwen-7B zero-shot 15.3%); on MMLU, scaling is the only lever.
- Effects survive scoring-artifact checks: a continuous numeric-presence metric on GSM8K, an
  extraction audit on full outputs (2–4% failure), and a full-precision (bf16) control on the Qwen
  ladder that reproduces and enlarges the MMLU penalty.

## Repository layout

```
run_experiments_local.py   # harness: models (incl. Gemma-2), prompts, extraction, scoring, stats, --aggregate
make_figures.py            # Figures 1–3 (auto-scales to the number of families)
make_appendix.py           # Appendix C tables (Wilson CIs + Holm-adjusted McNemar)
results/                   # per-example CSVs (Qwen + Llama + Gemma, single fixed set) + aggregates
figures/                   # generated figures
paper.pdf                  # the three-family write-up (matches the arXiv preprint)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt   # Apple-silicon Mac required for the local (MLX) models
```

## Reproduce

```bash
# run each family on the fixed evaluation set:
python run_experiments_local.py --models qwen-0.5b qwen-1.5b qwen-3b qwen-7b
python run_experiments_local.py --models llama-1b llama-3b llama-8b --tag llama
python run_experiments_local.py --models gemma-2b gemma-9b --tag gemma
# rebuild every table and figure from the saved per-example CSVs:
python run_experiments_local.py --aggregate
python make_figures.py
```

## AI assistance disclosure

Experimental code, statistical analysis, figure generation, and an initial manuscript draft were
produced with the assistance of an AI coding assistant; this is disclosed in the paper's
Acknowledgements. The author reviewed and verified the methods, results, and claims and takes full
responsibility for them.

## License

MIT — see [LICENSE](LICENSE).

# Chain-of-Thought's Double Edge Is Model-Specific: Evidence Across Three Model Families

A controlled study of when chain-of-thought (CoT) prompting helps and when it *hurts* — and how
that depends on the model family. This is **version 2** of an earlier two-family study, extended with
a third model family.

**Author:** Samir Samal (American High School)
**Builds on:** [prompting-vs-model-scaling](https://github.com/ssamalsamir/prompting-vs-model-scaling) (v1, two families)

> **Status:** v2 reuses all v1 data (Qwen2.5 and Llama-3) and adds **Gemma-2 (2B, 9B)** as a third
> family. The Gemma runs and the three-family figures, tables, and manuscript are being finalized;
> this README and the paper will be updated when the third family's results land.

## Question

When a single prompting technique (chain-of-thought) helps one model on a task, is it safe to assume
it helps another? We test this by sweeping multiple instruction-tuned model families across sizes and
four prompting strategies (zero-shot, few-shot, CoT, structured/JSON) on three task types — SST-2
(sentiment), MMLU (knowledge), GSM8K (math reasoning) — with 4-bit quantization held constant, run
locally via MLX. Two seeds (42, 43), n = 300 per condition (pooled n = 600), Wilson confidence
intervals, and Holm-adjusted McNemar tests.

## Families

| Family | Sizes |
|---|---|
| Qwen2.5-Instruct | 0.5B / 1.5B / 3B / 7B |
| Llama-3 | 1B / 3B / 8B |
| **Gemma-2 (new in v2)** | **2B / 9B** |

## Findings (v1, two families — v2 tests whether they hold with a third)

- **Primary — chain-of-thought's effect on knowledge is model-specific.** In Qwen2.5, CoT *degrades*
  MMLU by 6–17 pp (worse with scale); in Llama-3, CoT is neutral-to-helpful. v2 asks whether Gemma-2
  patterns with Qwen (hurts) or Llama (neutral) — i.e., whether this is a genuine split among families.
- **Confirming prior work** (Sprague et al., 2024): prompting substitutes for scale on reasoning but
  barely on knowledge — on GSM8K, CoT at a small size beats zero-shot at a much larger size; on MMLU,
  scaling is the only lever.
- Effects survive scoring-artifact checks (continuous numeric-presence metric; re-extraction from full
  outputs) and replicate across two seeds.

## Repository layout

```
run_experiments_local.py   # harness: models (incl. Gemma-2), prompts, extraction, scoring, stats, --aggregate
make_figures.py            # Figures 1–3 (auto-scales to the number of families)
make_appendix.py           # Appendix C tables (Wilson CIs + Holm-adjusted McNemar)
results/                   # per-example CSVs (Qwen + Llama from v1; Gemma added in v2) + aggregates
figures/                   # generated figures
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt   # Apple-silicon Mac required for the local (MLX) models
```

## Reproduce

```bash
# the third family (v1's Qwen/Llama CSVs are already in results/):
python run_experiments_local.py --models gemma-2b gemma-9b --tag gemma --seed 42
python run_experiments_local.py --models gemma-2b gemma-9b --tag gemma --seed 43
# combine all three families and rebuild tables + figures:
python run_experiments_local.py --aggregate
python make_figures.py
```

## AI assistance disclosure

Experimental code, statistical analysis, figure generation, and an initial manuscript draft were
produced with the assistance of an AI coding assistant; this is disclosed in the paper's
Acknowledgements.

## License

MIT — see [LICENSE](LICENSE).

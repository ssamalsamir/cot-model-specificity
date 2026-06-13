# Chain-of-Thought's Double Edge Is Model-Specific: Evidence Across Three Model Families

A controlled study of when chain-of-thought (CoT) prompting helps and when it *hurts* — and how
that depends on the model family. This is **version 3** of an earlier two-family study, extended with
a third model family now run at full two-seed parity.

**Author:** Samir Samal (American High School)
**Builds on:** [prompting-vs-model-scaling](https://github.com/ssamalsamir/prompting-vs-model-scaling) (v1, two families)

> **Status:** Complete. v3 reuses all v1 data (Qwen2.5 and Llama-3) and adds **Gemma-2 (2B, 9B)** as
> a third family at **two seeds (42 and 43), pooled n = 600** — matching the other families. The
> three-pattern CoT-on-knowledge finding replicates across both seeds. See [`paper.pdf`](paper.pdf)
> for the three-family write-up.

## Question

When a single prompting technique (chain-of-thought) helps one model on a task, is it safe to assume
it helps another? We test this by sweeping multiple instruction-tuned model families across sizes and
four prompting strategies (zero-shot, few-shot, CoT, structured/JSON) on three task types — SST-2
(sentiment), MMLU (knowledge), GSM8K (math reasoning) — with 4-bit quantization held constant, run
locally via MLX. All three families at two seeds (42, 43), n = 300 per condition per seed (pooled
n = 600), Wilson confidence intervals, and Holm-adjusted McNemar tests.

## Families

| Family | Sizes |
|---|---|
| Qwen2.5-Instruct | 0.5B / 1.5B / 3B / 7B |
| Llama-3 | 1B / 3B / 8B |
| **Gemma-2 (added as the third family)** | **2B / 9B** |

## Findings (three families, two seeds each)

- **Primary — chain-of-thought's effect on knowledge is model-specific, and three families show three
  distinct patterns.** Qwen2.5: CoT *degrades* MMLU increasingly with scale (−7 to −17 pp). Gemma-2:
  CoT harms the small model but the harm *fades* with scale (−15 pp at 2B → −1 pp at 9B; with n = 600
  the 2B harm is significant at *p* < 0.001 while the 9B effect is not significant). Llama-3: CoT
  is neutral-to-helpful. Not just *whether* CoT hurts knowledge, but *how that changes with scale*, is
  family-dependent.
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
results/                   # per-example CSVs (Qwen + Llama from v1; Gemma at seeds 42+43) + aggregates
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

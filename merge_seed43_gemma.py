"""Merge the seed-43 Gemma chunk CSVs into one canonical results_seed_43_gemma.csv.

These per-example CSVs carry NO per-row identifier column, so de-duplication is
unsafe (it would collapse a 300-row condition to one row). Instead we simply
concatenate the per-chunk files — each chunk owns a disjoint (model, benchmark)
group — and assert exactly 300 rows per (model, benchmark, strategy) condition.
"""
import pandas as pd, glob, sys
from collections import Counter

CHUNKS = sorted(glob.glob("results/results_seed_43_g43_*.csv"))
EXPECT_MODELS = ["gemma-2b", "gemma-9b"]
EXPECT_TASKS = ["sst2", "gsm8k", "mmlu"]
EXPECT_STRATS = ["zero_shot", "few_shot", "cot", "structured"]

if not CHUNKS:
    sys.exit("No chunk files found.")
print("Merging chunks:")
for f in CHUNKS:
    print("  -", f)

df = pd.concat([pd.read_csv(f) for f in CHUNKS], ignore_index=True)
c = Counter(zip(df.model, df.benchmark, df.strategy))

missing, wrong = [], []
for m in EXPECT_MODELS:
    for b in EXPECT_TASKS:
        for s in EXPECT_STRATS:
            n = c.get((m, b, s), 0)
            if n == 0:
                missing.append((m, b, s))
            elif n != 300:
                wrong.append((m, b, s, n))

print(f"\nTotal rows: {len(df)}  (expect 7200)")
print(f"Conditions present: {len(c)}/24")
if missing:
    print("MISSING conditions:", missing)
if wrong:
    print("WRONG-count conditions:", wrong)
if missing or wrong or len(df) != 7200:
    sys.exit("\nABORT: coverage check failed — not writing canonical file.")

df.to_csv("results/results_seed_43_gemma.csv", index=False)
print("\nOK — wrote results/results_seed_43_gemma.csv with all 24 conditions x 300.")

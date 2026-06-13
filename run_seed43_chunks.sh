#!/bin/zsh
# Finish the seed-43 Gemma run in crash-safe chunks (one model x one benchmark each).
# Prior full-run attempts died to machine sleep; each chunk here writes its own
# tagged CSV on completion, so a crash costs at most one chunk.
cd "$HOME/cot-model-specificity"
source "$HOME/.venvs/promptscaling/bin/activate"

check_complete() {
  python -c "import csv,sys; n=sum(1 for _ in csv.DictReader(open(sys.argv[1]))); sys.exit(0 if n>=1200 else 1)" "$1" 2>/dev/null
}

run_chunk() {
  local model=$1 bench=$2 tag=$3
  local csv="results/results_seed_43_${tag}.csv"
  if check_complete "$csv"; then
    echo "[chunks] $tag already complete, skipping"
    return 0
  fi
  for attempt in 1 2 3; do
    echo "[chunks] running $tag (attempt $attempt) at $(date)"
    python run_experiments_local.py --models "$model" --benchmarks "$bench" \
      --seed 43 --tag "$tag" >> "chunk_${tag}.log" 2>&1
    if check_complete "$csv"; then
      echo "[chunks] $tag done at $(date)"
      return 0
    fi
  done
  echo "[chunks] $tag FAILED after 3 attempts"
  return 1
}

fail=0
run_chunk gemma-2b mmlu  g43_2b_mmlu  || fail=1
run_chunk gemma-2b gsm8k g43_2b_gsm8k || fail=1
run_chunk gemma-9b sst2  g43_9b_sst2  || fail=1
run_chunk gemma-9b mmlu  g43_9b_mmlu  || fail=1
run_chunk gemma-9b gsm8k g43_9b_gsm8k || fail=1
echo "[chunks] ALL FINISHED fail=$fail at $(date)"
exit $fail

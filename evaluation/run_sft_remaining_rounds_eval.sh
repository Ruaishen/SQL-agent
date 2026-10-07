#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/sqlagent/sql-agent-rl
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
RUN_ID=${1:-$(date -u +%Y%m%dT%H%M%SZ)}
ROOT=artifacts/sql_planner/reasoning_dpo_strict_631_v1
DATASET="$ROOT/chosen_631_all_turns_sft"
BATCH_SIZE=${BATCH_SIZE:-64}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
started=$(date +%s)
trap 'code=$?; echo "PIPELINE_EXIT_CODE=$code PIPELINE_ENDED_UTC=$(date -u +%FT%TZ) TOTAL_SECONDS=$(($(date +%s)-started))"' EXIT
echo "PIPELINE_STARTED_UTC=$(date -u +%FT%TZ) RUN_ID=$RUN_ID"
echo "BATCH_SIZE=$BATCH_SIZE MAX_NUM_SEQS=$BATCH_SIZE GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
for variant in sft chosen_sft; do
  if [[ "$variant" == sft ]]; then
    run_root="$ROOT/sequential_base_fp32_2epoch/sft"
  else
    run_root="$ROOT/chosen_631_all_turns_sft_continue_2epoch"
  fi
  stage_started=$(date +%s)
  echo "STAGE=$variant START=$(date -u +%FT%TZ) MODEL=$run_root/checkpoint_epoch_2"
  python -u -m evaluation.run_reasoning_sft \
    --model "$run_root/checkpoint_epoch_2" \
    --dataset "$DATASET" \
    --tasks data/external_dev.jsonl \
    --env-config configs/env_sql_planner_qwen25_coder_3b.yaml \
    --spider-root /root/autodl-tmp/sqlagent/datasets/spider/spider_data \
    --output-dir "$run_root/eval_epoch_2_remaining_rounds_v1_$RUN_ID" \
    --batch-size "$BATCH_SIZE" --max-tokens 512 \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-num-seqs "$BATCH_SIZE"
  echo "STAGE=$variant END=$(date -u +%FT%TZ) SECONDS=$(($(date +%s)-stage_started))"
done
python -u - "$ROOT" "$RUN_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_id = sys.argv[2]
report = {}
for name, relative in (
    ("sft", "sequential_base_fp32_2epoch/sft"),
    ("chosen_sft", "chosen_631_all_turns_sft_continue_2epoch"),
):
    output = root / relative / f"eval_epoch_2_remaining_rounds_v1_{run_id}"
    result = json.loads((output / "summary.json").read_text())
    if result["completed"] != result["total"] or result["total"] != 1034:
        raise RuntimeError(f"Incomplete evaluation: {name}")
    report[name] = {"output_dir": str(output), **result}
report["accuracy_delta"] = report["chosen_sft"]["execution_accuracy"] - report["sft"]["execution_accuracy"]
destination = root / "reports" / f"remaining_rounds_v1_{run_id}.json"
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False), flush=True)
PY

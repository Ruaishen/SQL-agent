#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/sqlagent/sql-agent-rl
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
CONFIG=configs/sft_chosen_631_all_turns_continue_2epoch.yaml
RUN_ROOT=artifacts/sql_planner/reasoning_dpo_strict_631_v1/chosen_631_all_turns_sft_continue_2epoch
DATASET=artifacts/sql_planner/reasoning_dpo_strict_631_v1/chosen_631_all_turns_sft
started=$(date +%s)
trap 'code=$?; echo "PIPELINE_EXIT_CODE=$code PIPELINE_ENDED_UTC=$(date -u +%FT%TZ) TOTAL_SECONDS=$(($(date +%s)-started))"' EXIT
echo "PIPELINE_STARTED_UTC=$(date -u +%FT%TZ)"
echo "SUPERVISION=all_assistant_turns TRAJECTORIES=631 EPOCHS=2"
echo "STAGE=sft_train START=$(date -u +%FT%TZ)"
python -u -m sft.train --config "$CONFIG"
echo "STAGE=sft_train END=$(date -u +%FT%TZ)"
echo "STAGE=eval_epoch_2 START=$(date -u +%FT%TZ)"
python -u -m evaluation.run_reasoning_sft \
  --model "$RUN_ROOT/checkpoint_epoch_2" \
  --dataset "$DATASET" \
  --tasks data/external_dev.jsonl \
  --env-config configs/env_sql_planner_qwen25_coder_3b.yaml \
  --spider-root /root/autodl-tmp/sqlagent/datasets/spider/spider_data \
  --output-dir "$RUN_ROOT/eval_epoch_2" \
  --batch-size 8 --max-tokens 512 \
  --gpu-memory-utilization 0.82 --max-num-seqs 8
echo "STAGE=eval_epoch_2 END=$(date -u +%FT%TZ)"
python -u - <<'PY'
import json
from pathlib import Path

root = Path('artifacts/sql_planner/reasoning_dpo_strict_631_v1')
run = root / 'chosen_631_all_turns_sft_continue_2epoch'
training = json.loads((run / 'run_summary.json').read_text())
result = json.loads((run / 'eval_epoch_2/summary.json').read_text())
baseline = json.loads((root / 'sequential_base_fp32_2epoch/sft/eval_epoch_2/summary.json').read_text())
if training['completed_steps'] != training['configured_steps']:
    raise RuntimeError('Chosen SFT training incomplete')
if result['completed'] != result['total'] or result['total'] != 1034:
    raise RuntimeError('Chosen SFT evaluation incomplete')
report = {
    'status': 'completed',
    'epochs': 2,
    'initial_checkpoint': training['config']['student_model'],
    'checkpoint': training['checkpoint'],
    'supervision': 'all_assistant_turns',
    'baseline_correct': baseline['correct'],
    'correct': result['correct'],
    'total': result['total'],
    'execution_accuracy': result['execution_accuracy'],
    'accuracy_delta': result['execution_accuracy'] - baseline['execution_accuracy'],
    'status_counts': result['status_counts'],
}
(run / 'continuation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
PY

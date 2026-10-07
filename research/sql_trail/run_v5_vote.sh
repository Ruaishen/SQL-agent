#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/sqlagent/sql-agent-rl
export PYTHONUNBUFFERED=1
output_dir=${1:?Pass a new output directory for this evaluation}
mkdir -p "$output_dir"
exec >> "$output_dir/chain.log" 2>&1
date -u
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/Qwen2.5-Coder-3B-Instruct \
  --host 127.0.0.1 --port 8004 --max-model-len 16384 \
  --gpu-memory-utilization 0.75 > "$output_dir/vllm.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in $(seq 1 180); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo 'Model server exited; inspect vllm.log'
    exit 1
  fi
  if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8004/health >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -ne 1 ]; then
  echo 'Model server readiness timed out'
  exit 1
fi
python -m evaluation.run_v5_voting \
  --endpoint http://127.0.0.1:8004 \
  --model /root/autodl-tmp/Qwen2.5-Coder-3B-Instruct \
  --task-file data/external_dev.jsonl \
  --env-config configs/env.yaml \
  --spider-root /root/autodl-tmp/sqlagent/datasets/spider/spider_data \
  --source-dir artifacts/sql_planner/qwen25_coder_3b_base_direct_seeded_evidence_columns_v5_eval \
  --output-dir "$output_dir" --samples 8 --workers 16
echo 'Eight-sample voting evaluation completed'
date -u

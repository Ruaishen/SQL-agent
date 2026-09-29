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
python -m evaluation.run_tool_xml \
  --endpoint http://127.0.0.1:8004 \
  --model /root/autodl-tmp/Qwen2.5-Coder-3B-Instruct \
  --task-file data/external_dev.jsonl \
  --env-config configs/env_sql_planner_qwen25_coder_3b.yaml \
  --spider-root /root/autodl-tmp/sqlagent/datasets/spider/spider_data \
  --single-turn \
  --output-dir "$output_dir" --workers 8 --max-tokens 512
PYTHONPATH=. python research/sql_trail/audit.py \
  --multi-run "$output_dir" --output-dir "$output_dir/rescore"
echo 'Evaluation and rescoring completed'
date -u

"""Score generated final SQL tokens using the full, unscaled model distribution."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path


def sql_token_indices(tokenizer, token_ids, sql):
    """Select generated tokens overlapping the JSON-escaped SQL string contents."""
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    spans = []
    pattern = re.compile(r'"sql"\s*:\s*("(?:[^"\\]|\\.)*")')
    for match in pattern.finditer(text, max(0, text.rfind("<tool>"))):
        if json.loads(match.group(1)) == sql:
            spans.append((match.start(1) + 1, match.end(1) - 1))
    if len(spans) != 1:
        raise ValueError(f"Expected one matching SQL string; found {len(spans)}")
    start, end = spans[0]
    boundaries = [0] + [len(tokenizer.decode(token_ids[:i], skip_special_tokens=True))
                        for i in range(1, len(token_ids) + 1)]
    indices = [i for i in range(len(token_ids))
               if boundaries[i] < end and boundaries[i + 1] > start]
    if not indices:
        raise ValueError("SQL has no generated tokens")
    return indices


def entropy_from_logits(logits):
    import torch
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def final_sql_turn(record):
    if not record.get("final_sql"):
        return None
    candidates = [t for t in record["turns"]
                  if t.get("arguments", {}).get("sql") == record["final_sql"]
                  and "generated_token_ids" in t]
    return candidates[-1] if candidates else None


def score_sql_turn(model, tokenizer, turn, sql, chunk_size=32):
    import torch
    generated = turn["generated_token_ids"]
    prompt = turn["prompt_token_ids"]
    indices = sql_token_indices(tokenizer, generated, sql)
    inputs = torch.tensor([prompt + generated[:-1]], dtype=torch.long, device="cuda")
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0]
    positions = torch.tensor([len(prompt) + i - 1 for i in indices], device="cuda")
    selected = hidden.index_select(0, positions)
    del hidden
    entropies = []
    for chunk in selected.split(chunk_size):
        entropies.extend(entropy_from_logits(model.lm_head(chunk)).cpu().tolist())
    return dict(sql_token_count=len(indices), generated_token_indices=indices,
                sql_token_ids=[generated[i] for i in indices], token_entropy_nats=entropies,
                entropy_sum_nats=sum(entropies),
                mean_token_entropy_nats=sum(entropies) / len(entropies))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--logit-chunk-size", type=int, default=32)
    args = parser.parse_args()
    if args.logit_chunk_size < 1:
        parser.error("Chunk size must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    start = time.perf_counter()
    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    assert "memory" not in manifest and manifest["record_token_ids"]
    tokenizer = AutoTokenizer.from_pretrained(manifest["model"])
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    output_path = args.run_dir / "sql_token_entropy.jsonl"
    results = [json.loads(line) for line in output_path.read_text().splitlines()] if output_path.exists() else []
    done = {r["task_id"] for r in results}
    records = [json.loads(p.read_text()) for p in sorted((args.run_dir / "trajectories").glob("*.json"))]
    assert len(records) == 1034
    with torch.inference_mode(), output_path.open("a") as output:
        for record in records:
            if record["task_id"] in done:
                continue
            result = {k: record[k] for k in ("task_id", "difficulty", "correct", "status", "final_sql")}
            turn = final_sql_turn(record)
            if turn is None:
                result.update(sql_token_count=0, excluded_reason="No model-generated final SQL")
            else:
                result.update(score_sql_turn(model, tokenizer, turn, record["final_sql"],
                                             args.logit_chunk_size))
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            results.append(result)
            if len(results) % 100 == 0:
                print(json.dumps({"entropy_completed": len(results), "elapsed_seconds": time.perf_counter() - start}), flush=True)
    summary = {
        "method": "Full vocabulary Shannon entropy of original model logits, temperature=1, teacher-forced on exact generated token IDs and original prompt IDs",
        "scope": "Tokens overlapping final SQL JSON string contents; reasoning, tool wrapper and observations excluded; boundary tokens counted once",
        "forced_submission_policy": "Use last model-generated occurrence of the final SQL; evaluator-generated fallback text excluded",
        "scorer": "ExecutionVerifier", "total_tasks": len(records),
        "execution_accuracy": sum(r["correct"] is True for r in records) / len(records),
        "excluded_tasks": [r["task_id"] for r in results if not r["sql_token_count"]],
        "groups": {}, "entropy_wall_seconds": time.perf_counter() - start,
        "numerical_precision": "BF16 model forward; FP32 log_softmax and entropy reduction; HF SDPA replay may differ slightly from vLLM kernels",
    }
    for name, correct in [("correct_sql", True), ("incorrect_sql", False)]:
        group = [r for r in results if r["correct"] is correct and r["sql_token_count"]]
        tokens = sum(r["sql_token_count"] for r in group)
        entropy = sum(r["entropy_sum_nats"] for r in group)
        summary["groups"][name] = {
            "tasks_with_sql": len(group), "sql_token_count": tokens,
            "mean_per_token_entropy_nats": entropy / tokens if tokens else None,
            "mean_per_token_entropy_bits": entropy / tokens / math.log(2) if tokens else None,
            "mean_per_sql_token_entropy_nats": sum(r["mean_token_entropy_nats"] for r in group) / len(group) if group else None,
        }
    (args.run_dir / "entropy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

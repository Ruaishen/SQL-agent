# DPO pairs from all incorrect reasoning trajectories

> Historical protocol: the neutralized-reasoning and full-eligible dataset described
> below belong to an earlier experiment. The current strict run preserves raw
> reasoning and retains only blind-audit accept pairs: 631 pairs after at most two
> additional attempts per previously unaccepted task. Use the audited JSONL preparation
> commands in [the project README](../README.md#dpo数学目标与当前偏好数据构造).
> The current DPO/GRPO configs point to `reasoning_dpo_strict_631_v1`; the old training
> commands below require explicitly configuring the historical dataset paths.

The source is the completed 7,000-record `artifacts/sql_planner/reasoning`
collection. `dpo.pairs select` accounts for every incorrect trajectory without
changing source files. It validates the Spider gold query on the actual database
before declaring a record eligible. A failed gold validation remains in the
selection manifest with `block_reason=gold_not_executable`; it must never be
used as an unverified chosen example.

The fork is selected in this order:

1. After the **last successful execute_sql observation whose SQL is verified
   wrong** against the gold result.
2. After a failed `execute_sql` observation, if no verified wrong success exists.
3. After another successful SQL observation, if there is no verified wrong
   execution; this can include a correct intermediate SQL followed by an
   incorrect final submission.
4. Before an untested final submission or before the malformed assistant
   response that ended an `invalid_format` trajectory.

`fork_reason` is saved in each selection entry and preference pair. The rejected
branch is the original source continuation. DeepSeek receives a teacher-only
gold SQL hint and generates the chosen continuation. The student-facing chosen
branch contains the original prompt and real replayed observations, but never
the teacher hint. The chosen submission must pass full-result verification.
The teacher may execute that SQL first or submit it directly after the fork.
Both are verified against the real database. Direct submissions are marked
`chosen_tested_before_submit=false`; the generation report counts both modes.
If teacher reasoning explicitly mentions the privileged hint or target SQL,
that turn's reasoning is replaced with a neutral action description. The SQL,
tool action, and observation stay unchanged. `sanitized_reasoning_turns` records
every edited turn. Run `python -m dpo.sanitize` to apply the same check to a
completed batch and rebuild pairs from the saved attempts.

## Offline selection

Run in the project checkout with the local Spider files and tokenizer path
configured in `configs/env_sql_planner_local_multi_table_v11.yaml`:

```bash
python -m dpo.pairs select
```

This writes `artifacts/sql_planner/reasoning_dpo_all_errors_v1/selection.json`.
The manifest reports all incorrect trajectories, eligible records, blocked
records, and counts by fork reason. It is deterministic for an unchanged source,
gold dataset, and environment config.

For the current snapshot: 1,907 incorrect trajectories are accounted for;
1,904 are eligible and 3 have an unexecutable gold query. Eligible fork counts
are 1,832 after a verified wrong SQL success, 45 after another SQL success,
14 before an untested submission, and 13 before an invalid-format response.
The completed batch contains 1,904 pairs: 1,063 tested the chosen SQL before
submission and 841 submitted it directly. Hint language was neutralized in
1,407 pairs (2,170 reasoning turns); this editing is recorded in each pair.

## Generate chosen branches after the API key is available

Set `DEEPSEEK_API_KEY` in the process environment and run:

```bash
python -m dpo.pairs generate --limit 10
python -m dpo.pairs generate --workers 4
```

The first command is an optional small batch. The second skips existing pairs.
Each teacher attempt is saved under `attempts/`; only validated preferences are
saved under `pairs/`. Review failures before tokenization. The key is not
written into output files. These are synthetic, gold-guided preferences and
carry `teacher_used_gold_hint=true`.

The current teacher system prompt is identical to the ordinary agent prompt.
Only the post-fork user instruction supplies gold SQL, forbids mentioning the
hint in reasoning, and requires claimed observations to come from real tool
returns. Response retries refer back to this instruction without repeating
gold SQL. The ordinary system prompt's existing prohibition on gold SQL is
preserved verbatim. Earlier generated datasets retain their original protocol.

To repeat the previous 100-task audit with Flash high and preserve raw reasoning:

```bash
python -m dpo.compare --variants flash_high \
  --sample-manifest artifacts/sql_planner/dpo_chosen_hint_compare_100_v2/manifest.json \
  --output-dir artifacts/sql_planner/dpo_flash_high_base_system_100_v3 --workers 6
```

## Tokenize and train on the GPU machine

```bash
python -m dpo.prepare \
  --pairs-dir artifacts/sql_planner/reasoning_dpo_all_errors_v1/pairs \
  --model artifacts/sql_planner/reasoning_balanced_2000_500_qwen25_coder_3b_sft/checkpoint_epoch_2 \
  --output-dir artifacts/sql_planner/reasoning_dpo_all_errors_v1/tokenized
python -m dpo.train score-reference --config configs/dpo_reasoning_sql_success.yaml
python -m dpo.train train --config configs/dpo_reasoning_sql_success.yaml
```

Adjust YAML paths to the GPU machine. The DPO loss mask includes only assistant
tokens from each branch's fork onward. It excludes the shared prefix, system
message, user question, and tool observations. Each branch is scored under its
own actual observation history. Reference log probabilities are cached before
full-parameter policy training, so the reference and policy models need not
occupy GPU memory simultaneously.

The resulting checkpoint is configured as the GRPO starting model in
`configs/grpo_reasoning_tool_after_dpo.yaml`. Report evaluation by `fork_reason`:
the successful-but-wrong SQL group measures the original no-error-feedback
problem, while fallback groups cover SQL errors, untested submissions, and
format failures.

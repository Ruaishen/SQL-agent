# Spider CoT GRPO: standard and conditional Gold injection

The two configs use the same `reasoning_tool` protocol, train pool, initial model,
sampling settings, binary execution verifier and optimizer. Both use a
clip-higher interval of [0.8, 1.28], no KL regularization, temperature 1.0,
top-p 0.99, no top-k filter, and learning rate 1e-6. Only the Gold
injection switch differs. Point **both** `student_model` and `reference_model`
in the configs to the same Spider CoT SFT checkpoint before preparing the runs.
The checked-in checkpoint path follows `sft_reasoning_balanced_2000_500.yaml`;
verify it on the training host or replace it with the newer Spider7 CoT SFT checkpoint.

For each prompt the policy first samples six full trajectories without Gold.
The standard run applies ordinary GRPO. In the injection run, an all-wrong
group triggers up to three more attempts with the **same frozen rollout policy**
and a training-only Gold SQL hint. At most one successful guided trajectory is
appended. It must list tables, inspect schema, successfully execute the exact
SQL it submits, and pass the normal verifier. The hint is removed from all
training contexts and saved trajectory tensors. The old policy re-scores the
guided assistant tokens on the normal hint-free context. Every trajectory in
the resulting group uses the same clipped GRPO surrogate. This is a deliberate
mixed-context demonstration objective, not an unbiased on-policy estimator.

Prepare and train both runs independently, using the same environment config:

```bash
python -m grpo.prepare --config configs/grpo_reasoning_tool_standard.yaml
python -m grpo.prepare --config configs/grpo_reasoning_tool_gold_injection.yaml
python -m grpo.train --config configs/grpo_reasoning_tool_standard.yaml --env-config configs/env_sql_planner_qwen25_coder_3b.yaml --max-steps 128
python -m grpo.train --config configs/grpo_reasoning_tool_gold_injection.yaml --env-config configs/env_sql_planner_qwen25_coder_3b.yaml --max-steps 128
```

The two prepare commands select the same task IDs because their selection
parameters and seed match. Compare the prepared `task_pool.jsonl` hashes before
the runs. The repository's `evaluation/run_reasoning_sft.py` evaluates both
checkpoints with the same prompt and held-out tasks; pass its CoT SFT dataset
manifest via `--dataset`, use the same `--tasks`, and set `--max-tokens 2048`.
Use separate output directories for the two evaluations. The normal sampling
budget is identical (six trajectories per prompt); the injection run consumes
additional Gold-guided attempts on all-wrong prompts, reported in its metrics.

Each training step records `original_student_success_rate`,
`original_student_correct_count_histogram`, `original_all_zero_groups`,
`gold_injected_groups`, and `gold_injection_attempts`. The ordinary group
reward statistics include the inserted demonstration; use the original-student
fields when judging the policy's own success. `policy_clip_fraction` counts
nonzero-advantage assistant tokens whose clipped surrogate actually has zero
policy gradient. `on_policy_policy_clip_fraction` and
`gold_guided_policy_clip_fraction` split the two sources. The separate `clip`
and `*_ratio_outside_fraction` fields count ratios outside the interval,
including tokens whose policy gradient is still active. With `kl_beta: 0.0`,
the reference model is not loaded or scored and the KL metric is zero.

The implementation takes one optimizer step per newly sampled batch, so the
policy clip fraction is expected to start near zero. Check the logged counts
before interpreting the fractions. A complete comparison uses the same
held-out execution accuracy and inference budget for both checkpoints.

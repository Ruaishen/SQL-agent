# Spider CoT binary execution GRPO

Use `configs/grpo_reasoning_tool_standard.yaml` to initialize from the CoT SFT
checkpoint.
For every question the current policy samples six independent trajectories.
The final submitted SQL receives reward 1 if its execution result matches the
reference SQL's result, otherwise 0. Group-relative advantages are computed from
these six rewards. All-correct and all-wrong groups have zero policy advantage;
they are kept in the reported statistics. No Gold-guided generation or repair is
performed. Reference SQL is used only by the execution verifier.

The current configs use clip-higher with ratio interval [0.8, 1.28], no KL
regularization, temperature 1.0, top-p 0.99, no top-k filter, and learning rate
1e-6. They retain four questions per step, ten exploratory actions followed by
final submission, and 128 training steps. The reference model is not loaded or
scored with `kl_beta: 0.0`.

```bash
python -m grpo.prepare --config configs/grpo_reasoning_tool_standard.yaml
python -m grpo.train --config configs/grpo_reasoning_tool_standard.yaml --env-config configs/env.yaml --max-steps 128
```

Verify the model, data, and output paths on the training host before preparing.
Use a fresh output directory for a new run. Enabling Gold injection is rejected.
SFT data repair remains separate from this RL rollout path.

Training logs include execution accuracy, all-zero/all-one/mixed group counts,
and `policy_clip_fraction`. The latter counts nonzero-advantage assistant tokens
whose clipped surrogate has zero policy gradient. `on_policy_policy_clip_fraction`
reports the same source-specific statistic. `clip` and
`on_policy_ratio_outside_fraction` instead count ratios outside the interval,
including tokens whose policy gradient is still active. One optimizer step is
taken per freshly sampled batch, so clipping may initially be near zero.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class GrpoLossOutput:
    token_loss: Tensor
    policy_loss: Tensor
    kl: Tensor
    ratio: Tensor
    clipped: Tensor
    policy_clipped: Tensor


def grpo_token_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor | None,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_ratio: float,
    clip_ratio_high: float | None = None,
    kl_beta: float,
) -> GrpoLossOutput:
    shape = current_log_probs.shape
    if current_log_probs.ndim != 2 or any(
        tensor.shape != shape for tensor in (old_log_probs, action_mask)
    ):
        raise ValueError("GRPO token tensors must share [batch, response_length] shape")
    if reference_log_probs is not None and reference_log_probs.shape != shape:
        raise ValueError("reference log probabilities must share [batch, response_length] shape")
    if advantages.shape != (shape[0],):
        raise ValueError("advantages must have shape [batch]")
    high = clip_ratio if clip_ratio_high is None else clip_ratio_high
    if not 0 <= clip_ratio <= high < 1 or kl_beta < 0:
        raise ValueError("invalid clip_ratio or kl_beta")
    if kl_beta > 0 and reference_log_probs is None:
        raise ValueError("reference log probabilities required when kl_beta > 0")
    mask = action_mask.bool()
    if not mask.any():
        raise ValueError("action_mask contains no trainable tokens")
    checked = [("current", current_log_probs), ("old", old_log_probs)]
    if kl_beta > 0:
        checked.append(("reference", reference_log_probs))
    for name, tensor in checked:
        if not torch.isfinite(tensor[mask]).all():
            raise ValueError(f"{name} log probabilities contain non-finite values")
    old = old_log_probs.detach()
    advantage = advantages.detach().unsqueeze(1)
    log_ratio = torch.clamp(current_log_probs - old, min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + high)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -surrogate
    if kl_beta > 0:
        ref_minus_current = torch.clamp(
            reference_log_probs.detach() - current_log_probs, min=-20.0, max=20.0
        )
        kl = ref_minus_current.exp() - ref_minus_current - 1.0
        token_loss = policy_loss + kl_beta * kl
    else:
        kl = torch.zeros_like(policy_loss)
        token_loss = policy_loss
    clipped = (ratio != clipped_ratio).to(current_log_probs.dtype)
    policy_clipped = (
        ((advantage > 0) & (ratio > 1.0 + high))
        | ((advantage < 0) & (ratio < 1.0 - clip_ratio))
    ).to(current_log_probs.dtype)
    return GrpoLossOutput(token_loss, policy_loss, kl, ratio, clipped, policy_clipped)

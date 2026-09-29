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


def grpo_token_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    advantages: Tensor,
    action_mask: Tensor,
    *,
    clip_ratio: float,
    kl_beta: float,
) -> GrpoLossOutput:
    shape = current_log_probs.shape
    if current_log_probs.ndim != 2 or any(
        tensor.shape != shape for tensor in (old_log_probs, reference_log_probs, action_mask)
    ):
        raise ValueError("GRPO token tensors must share [batch, response_length] shape")
    if advantages.shape != (shape[0],):
        raise ValueError("advantages must have shape [batch]")
    if not 0 <= clip_ratio < 1 or kl_beta < 0:
        raise ValueError("invalid clip_ratio or kl_beta")
    mask = action_mask.bool()
    if not mask.any():
        raise ValueError("action_mask contains no trainable tokens")
    for name, tensor in (
        ("current", current_log_probs),
        ("old", old_log_probs),
        ("reference", reference_log_probs),
    ):
        if not torch.isfinite(tensor[mask]).all():
            raise ValueError(f"{name} log probabilities contain non-finite values")
    old = old_log_probs.detach()
    reference = reference_log_probs.detach()
    advantage = advantages.detach().unsqueeze(1)
    log_ratio = torch.clamp(current_log_probs - old, min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    policy_loss = -surrogate
    ref_minus_current = torch.clamp(reference - current_log_probs, min=-20.0, max=20.0)
    kl = ref_minus_current.exp() - ref_minus_current - 1.0
    token_loss = policy_loss + kl_beta * kl
    clipped = (ratio != clipped_ratio).to(current_log_probs.dtype)
    return GrpoLossOutput(token_loss, policy_loss, kl, ratio, clipped)

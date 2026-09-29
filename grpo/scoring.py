from __future__ import annotations

import torch
from torch import Tensor


def causal_selected_log_probs(logits: Tensor, input_ids: Tensor) -> Tensor:
    """Return log p(input_ids[t] | input_ids[:t]) for positions 1..L-1."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [B,L,V] and input_ids [B,L]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes differ")
    shifted_logits = logits[:, :-1, :].float()
    targets = input_ids[:, 1:].unsqueeze(-1)
    target_logits = shifted_logits.gather(dim=-1, index=targets).squeeze(-1)
    return target_logits - torch.logsumexp(shifted_logits, dim=-1)


def causal_selected_log_probs_chunked(
    logits: Tensor, input_ids: Tensor, *, vocab_chunk_size: int = 2048
) -> Tensor:
    """Memory-bounded FP32 selected-token log-probs for frozen-model scoring."""
    if logits.requires_grad:
        raise ValueError("chunked scoring is only valid for detached frozen-model logits")
    if logits.ndim != 3 or input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise ValueError("expected aligned logits [B,L,V] and input_ids [B,L]")
    shifted = logits[:, :-1, :]
    targets = input_ids[:, 1:].unsqueeze(-1)
    target_logits = shifted.gather(dim=-1, index=targets).squeeze(-1).float()
    maxima = shifted.max(dim=-1).values.float()
    exp_sum = torch.zeros_like(maxima)
    for start in range(0, shifted.shape[-1], vocab_chunk_size):
        chunk = shifted[..., start : start + vocab_chunk_size].float()
        exp_sum += torch.exp(chunk - maxima.unsqueeze(-1)).sum(dim=-1)
        del chunk
    return target_logits - maxima - torch.log(exp_sum)


def build_action_mask(
    response_mask: Tensor,
    response_ids: Tensor,
    special_token_ids: set[int] | frozenset[int],
) -> Tensor:
    """Keep assistant content while excluding observations, padding and protocol tokens."""
    if response_mask.shape != response_ids.shape:
        raise ValueError("response_mask and response_ids must have the same shape")
    mask = response_mask.bool().clone()
    if special_token_ids:
        special = torch.zeros_like(mask)
        for token_id in special_token_ids:
            special |= response_ids.eq(token_id)
        mask &= ~special
    return mask

"""Masked GRPO objective with group-relative advantages and a k3 reference penalty.

The reduction and clipping conventions follow OpenRLHF v0.6.3.
"""
from collections import defaultdict
import torch


def group_advantages(rewards, group_ids):
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    advantages = torch.empty_like(rewards, dtype=torch.float32)
    for indices in groups.values():
        values = rewards[indices].float()
        advantages[indices] = (values - values.mean()) / (values.std(correction=1) + 1e-9)
    return advantages


def sequence_mean(values, mask):
    return ((values * mask).sum(-1) / mask.sum(-1)).mean()


def grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, mask,
              *, clip_epsilon=0.2, kl_coefficient=0.01):
    # Select away environment positions BEFORE exponentiation as well as after:
    # their likelihoods are not actions and cannot overflow an unused branch.
    active = mask.bool()
    log_ratio = torch.where(active, log_probs.float() - old_log_probs.float(), 0.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = sequence_mean(-torch.minimum(unclipped, clipped), mask)
    ref_ratio = torch.where(active, ref_log_probs.float() - log_probs.float(), 0.0)
    # OpenRLHF k3: no VERL-specific +/-20 or +/-10 clipping, no k3+ STE.
    kl = sequence_mean(ref_ratio.exp() - 1 - ref_ratio, mask)
    loss = policy_loss + kl_coefficient * kl
    return loss, {"policy_loss": policy_loss.detach(), "kl": kl.detach(),
                  "clip_fraction": sequence_mean((unclipped != clipped).float(), mask).detach()}

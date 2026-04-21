from __future__ import annotations

import torch


def apply_random_module_mask(
    availability_mask: torch.Tensor,
    drop_prob: float = 0.15,
    ensure_at_least_one: bool = True,
) -> torch.Tensor:
    """
    Randomly mask available modules during training.

    Args:
        availability_mask: [B, M], 1 if available, 0 if unavailable
        drop_prob: probability of dropping each currently available module
        ensure_at_least_one: if True, make sure at least one originally available module remains

    Returns:
        effective_mask: [B, M]
    """
    if availability_mask.ndim != 2:
        raise ValueError(
            f"availability_mask must be 2D [B, M], got shape {tuple(availability_mask.shape)}"
        )

    original = availability_mask.float()
    random_keep = (torch.rand_like(original) > drop_prob).float()

    effective = original * random_keep

    if ensure_at_least_one:
        bsz = effective.shape[0]
        for i in range(bsz):
            original_available = torch.where(original[i] > 0)[0]
            if len(original_available) == 0:
                continue
            if effective[i].sum() == 0:
                chosen_idx = original_available[torch.randint(len(original_available), (1,)).item()]
                effective[i, chosen_idx] = 1.0

    return effective
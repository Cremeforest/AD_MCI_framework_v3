from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def cox_ph_loss(
    risk_scores: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """
    Negative Cox partial log-likelihood using Breslow-style handling.

    Args:
        risk_scores: [N]
        times: [N]
        events: [N], 1 if event observed, 0 if censored

    Returns:
        scalar loss
    """
    if risk_scores.ndim != 1:
        risk_scores = risk_scores.squeeze(-1)
    if times.ndim != 1:
        times = times.squeeze(-1)
    if events.ndim != 1:
        events = events.squeeze(-1)

    order = torch.argsort(times, descending=True)
    risk_sorted = risk_scores[order]
    events_sorted = events[order].float()

    log_cumsum_risk = torch.logcumsumexp(risk_sorted, dim=0)
    loglik = (risk_sorted - log_cumsum_risk) * events_sorted

    num_events = events_sorted.sum().clamp_min(1.0)
    loss = -loglik.sum() / num_events
    return loss


def binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim != 1:
        logits = logits.squeeze(-1)
    if targets.ndim != 1:
        targets = targets.squeeze(-1)

    targets = targets.float()

    if pos_weight is not None:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(logits, targets)


def combined_framework_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_survival: float = 1.0,
    lambda_event3y: float = 0.5,
    lambda_highrisk: float = 0.5,
    event3y_pos_weight: Optional[torch.Tensor] = None,
    highrisk_pos_weight: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    survival_loss = cox_ph_loss(
        risk_scores=outputs["risk_score"],
        times=batch["time"],
        events=batch["event"],
    )

    event3y_loss = binary_classification_loss(
        logits=outputs["event3y_logit"],
        targets=batch["event_3y"],
        pos_weight=event3y_pos_weight,
    )

    highrisk_loss = binary_classification_loss(
        logits=outputs["highrisk_logit"],
        targets=batch["highrisk"],
        pos_weight=highrisk_pos_weight,
    )

    total_loss = (
        lambda_survival * survival_loss
        + lambda_event3y * event3y_loss
        + lambda_highrisk * highrisk_loss
    )

    loss_dict = {
        "total_loss": float(total_loss.detach().cpu().item()),
        "survival_loss": float(survival_loss.detach().cpu().item()),
        "event3y_loss": float(event3y_loss.detach().cpu().item()),
        "highrisk_loss": float(highrisk_loss.detach().cpu().item()),
    }

    return total_loss, loss_dict
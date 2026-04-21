from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadConfig:
    input_dim: int = 32
    hidden_dim: int = 32
    dropout: float = 0.10


class SurvivalHead(nn.Module):
    """
    Cox-style survival head.
    Output is a scalar risk score (log-risk).
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClassificationHead(nn.Module):
    """
    Binary classification head.
    Output is a scalar logit.
    """

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
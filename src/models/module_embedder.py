from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModuleEmbedderConfig:
    input_dim: int
    embed_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.10


class ModuleEmbedder(nn.Module):
    """
    Encode one clinical module into a fixed-length embedding.

    Input:
        x: [batch_size, input_dim]

    Output:
        z: [batch_size, embed_dim]
    """

    def __init__(self, config: ModuleEmbedderConfig) -> None:
        super().__init__()
        self.config = config

        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = self.norm(z)
        return z
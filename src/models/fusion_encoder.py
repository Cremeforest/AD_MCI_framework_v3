from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionEncoderConfig:
    embed_dim: int = 32
    num_modules: int = 4
    num_heads: int = 4
    ff_hidden_dim: int = 64
    dropout: float = 0.10


class FusionEncoder(nn.Module):
    """
    Transformer-lite fusion encoder for module embeddings.

    Input:
        module_embeddings: [batch_size, num_modules, embed_dim]
        availability_mask: [batch_size, num_modules]
            1 = available
            0 = missing / masked

    Output:
        fused_repr: [batch_size, embed_dim]
        token_repr: [batch_size, num_modules, embed_dim]
        fusion_weights: [batch_size, num_modules]
    """

    def __init__(self, config: FusionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.module_type_embedding = nn.Parameter(
            torch.randn(config.num_modules, config.embed_dim) * 0.02
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ff_hidden_dim, config.embed_dim),
        )

        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        self.gate = nn.Linear(config.embed_dim, 1)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        module_embeddings: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            module_embeddings: [B, M, D]
            availability_mask: [B, M] with values 0/1

        Returns:
            fused_repr: [B, D]
            token_repr: [B, M, D]
            fusion_weights: [B, M]
        """
        if module_embeddings.dim() != 3:
            raise ValueError(
                f"module_embeddings must be 3D [B, M, D], got shape {tuple(module_embeddings.shape)}"
            )
        if availability_mask.dim() != 2:
            raise ValueError(
                f"availability_mask must be 2D [B, M], got shape {tuple(availability_mask.shape)}"
            )

        bsz, num_modules, embed_dim = module_embeddings.shape
        if num_modules != self.config.num_modules:
            raise ValueError(
                f"Expected num_modules={self.config.num_modules}, got {num_modules}"
            )
        if embed_dim != self.config.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.config.embed_dim}, got {embed_dim}"
            )

        availability_mask = availability_mask.float()
        key_padding_mask = availability_mask == 0  # True means ignore

        type_emb = self.module_type_embedding.unsqueeze(0).expand(bsz, -1, -1)
        x = module_embeddings + type_emb

        attn_out, _ = self.attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        gate_logits = self.gate(x).squeeze(-1)  # [B, M]
        gate_logits = gate_logits.masked_fill(availability_mask == 0, float("-inf"))

        fusion_weights = torch.softmax(gate_logits, dim=1)

        # handle pathological case where all modules are missing for a sample
        all_missing = (availability_mask.sum(dim=1) == 0)
        if all_missing.any():
            fusion_weights = fusion_weights.clone()
            fusion_weights[all_missing] = 0.0

        fused_repr = torch.sum(x * fusion_weights.unsqueeze(-1), dim=1)

        return fused_repr, x, fusion_weights
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.module_embedder import ModuleEmbedder, ModuleEmbedderConfig
from src.models.fusion_encoder import FusionEncoder, FusionEncoderConfig
from src.models.heads import ClassificationHead, HeadConfig, SurvivalHead


@dataclass
class FrameworkModelConfig:
    baseline_input_dim: int = 6
    structure_input_dim: int = 5
    state_input_dim: int = 5
    dynamics_input_dim: int = 8

    module_embed_dim: int = 32
    module_hidden_dim: int = 64
    fusion_heads: int = 4
    fusion_ff_hidden_dim: int = 64
    head_hidden_dim: int = 32
    dropout: float = 0.10


class ClinicalFrameworkModel(nn.Module):
    """
    Unified modular clinical AI framework.

    Inputs:
        baseline_x:  [B, baseline_input_dim]
        structure_x: [B, structure_input_dim]
        state_x:     [B, state_input_dim]
        dynamics_x:  [B, dynamics_input_dim]

        availability_mask: [B, 4]
            order = [baseline, structure, state, dynamics]
            1 = module available
            0 = module missing or intentionally masked

    Outputs:
        dict with:
            risk_score
            event3y_logit
            highrisk_logit
            patient_embedding
            module_embeddings
            token_embeddings
            fusion_weights
    """

    def __init__(self, config: FrameworkModelConfig) -> None:
        super().__init__()
        self.config = config

        self.baseline_embedder = ModuleEmbedder(
            ModuleEmbedderConfig(
                input_dim=config.baseline_input_dim,
                embed_dim=config.module_embed_dim,
                hidden_dim=config.module_hidden_dim,
                dropout=config.dropout,
            )
        )
        self.structure_embedder = ModuleEmbedder(
            ModuleEmbedderConfig(
                input_dim=config.structure_input_dim,
                embed_dim=config.module_embed_dim,
                hidden_dim=config.module_hidden_dim,
                dropout=config.dropout,
            )
        )
        self.state_embedder = ModuleEmbedder(
            ModuleEmbedderConfig(
                input_dim=config.state_input_dim,
                embed_dim=config.module_embed_dim,
                hidden_dim=config.module_hidden_dim,
                dropout=config.dropout,
            )
        )
        self.dynamics_embedder = ModuleEmbedder(
            ModuleEmbedderConfig(
                input_dim=config.dynamics_input_dim,
                embed_dim=config.module_embed_dim,
                hidden_dim=config.module_hidden_dim,
                dropout=config.dropout,
            )
        )

        self.fusion_encoder = FusionEncoder(
            FusionEncoderConfig(
                embed_dim=config.module_embed_dim,
                num_modules=4,
                num_heads=config.fusion_heads,
                ff_hidden_dim=config.fusion_ff_hidden_dim,
                dropout=config.dropout,
            )
        )

        head_cfg = HeadConfig(
            input_dim=config.module_embed_dim,
            hidden_dim=config.head_hidden_dim,
            dropout=config.dropout,
        )
        self.survival_head = SurvivalHead(head_cfg)
        self.event3y_head = ClassificationHead(head_cfg)
        self.highrisk_head = ClassificationHead(head_cfg)

    def forward(
        self,
        baseline_x: torch.Tensor,
        structure_x: torch.Tensor,
        state_x: torch.Tensor,
        dynamics_x: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if availability_mask.shape[1] != 4:
            raise ValueError(
                f"availability_mask must have shape [B, 4], got {tuple(availability_mask.shape)}"
            )

        baseline_z = self.baseline_embedder(baseline_x)
        structure_z = self.structure_embedder(structure_x)
        state_z = self.state_embedder(state_x)
        dynamics_z = self.dynamics_embedder(dynamics_x)

        module_embeddings = torch.stack(
            [baseline_z, structure_z, state_z, dynamics_z],
            dim=1,
        )  # [B, 4, D]

        fused_repr, token_repr, fusion_weights = self.fusion_encoder(
            module_embeddings=module_embeddings,
            availability_mask=availability_mask,
        )

        risk_score = self.survival_head(fused_repr)
        event3y_logit = self.event3y_head(fused_repr)
        highrisk_logit = self.highrisk_head(fused_repr)

        return {
            "risk_score": risk_score,
            "event3y_logit": event3y_logit,
            "highrisk_logit": highrisk_logit,
            "patient_embedding": fused_repr,
            "module_embeddings": module_embeddings,
            "token_embeddings": token_repr,
            "fusion_weights": fusion_weights,
        }

    @torch.no_grad()
    def encode_patient(
        self,
        baseline_x: torch.Tensor,
        structure_x: torch.Tensor,
        state_x: torch.Tensor,
        dynamics_x: torch.Tensor,
        availability_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method for downstream embedding extraction / clustering.
        """
        outputs = self.forward(
            baseline_x=baseline_x,
            structure_x=structure_x,
            state_x=state_x,
            dynamics_x=dynamics_x,
            availability_mask=availability_mask,
        )
        return outputs["patient_embedding"]
from __future__ import annotations

from typing import Literal

import torch
from torch import nn


PARAM_NAMES = (
    "LineDensity_per_mm",
    "w1",
    "Theta1_deg",
    "Theta2_deg",
    "Inc_Angle_deg",
)
PARAM_LOW = torch.tensor([100.0, 0.3, 1.0, 1.0, 0.0], dtype=torch.float32)
PARAM_HIGH = torch.tensor([1200.0, 0.7, 30.0, 30.0, 30.0], dtype=torch.float32)
WAVELENGTHS_NM = torch.arange(400.0, 1700.1, 10.0, dtype=torch.float32)


def normalize_physical(value: torch.Tensor) -> torch.Tensor:
    low = PARAM_LOW.to(device=value.device, dtype=value.dtype)
    high = PARAM_HIGH.to(device=value.device, dtype=value.dtype)
    return (value - low) / (high - low)


def denormalize_structure(structure_norm: torch.Tensor) -> torch.Tensor:
    low = PARAM_LOW[:4].to(device=structure_norm.device, dtype=structure_norm.dtype)
    high = PARAM_HIGH[:4].to(device=structure_norm.device, dtype=structure_norm.dtype)
    return low + structure_norm * (high - low)


def normalize_incidence(incidence_deg: torch.Tensor | float) -> torch.Tensor:
    value = torch.as_tensor(incidence_deg, dtype=torch.float32)
    return value / PARAM_HIGH[4]


def canonicalize_structure_norm(structure_norm: torch.Tensor) -> torch.Tensor:
    theta1 = structure_norm[..., 2]
    theta2 = structure_norm[..., 3]
    swap = theta1 > theta2
    return torch.stack(
        [
            structure_norm[..., 0],
            torch.where(swap, 1.0 - structure_norm[..., 1], structure_norm[..., 1]),
            torch.minimum(theta1, theta2),
            torch.maximum(theta1, theta2),
        ],
        dim=-1,
    )


def build_activation(name: Literal["relu", "gelu", "silu", "tanh"]) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[name]()


class ResidualBlock(nn.Module):
    """Residual block matching the bundled forward checkpoints."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        activation: Literal["relu", "gelu", "silu", "tanh"] = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        norm_1 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        norm_2 = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            norm_1,
            build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            norm_2,
        )
        self.out_activation = build_activation(activation)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.out_activation(value + self.net(value))


class ForwardNet(nn.Module):
    """Normalized five-parameter input to a 131-point efficiency spectrum."""

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 131,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        *,
        activation: Literal["relu", "gelu", "silu", "tanh"] = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        output_activation: Literal["none", "sigmoid"] = "none",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_activation = output_activation

        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            build_activation(activation),
        ]
        for _ in range(num_blocks):
            layers.append(
                ResidualBlock(
                    hidden_dim,
                    activation=activation,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            )
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.net(value)
        return torch.sigmoid(output) if self.output_activation == "sigmoid" else output


class MultiCandidateTargetInverseNet(nn.Module):
    """Target spectrum plus normalized incidence to four normalized structures."""

    def __init__(
        self,
        spectrum_dim: int = 131,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.0,
        num_candidates: int = 4,
    ) -> None:
        super().__init__()
        self.num_candidates = num_candidates

        layers: list[nn.Module] = [nn.Linear(spectrum_dim + 1, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        layers.extend([nn.Linear(hidden_dim, num_candidates * 4), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, target: torch.Tensor, incidence_norm: torch.Tensor) -> torch.Tensor:
        if incidence_norm.ndim == 1:
            incidence_norm = incidence_norm[:, None]
        value = self.net(torch.cat([target, incidence_norm], dim=1))
        return canonicalize_structure_norm(value.reshape(len(target), self.num_candidates, 4))


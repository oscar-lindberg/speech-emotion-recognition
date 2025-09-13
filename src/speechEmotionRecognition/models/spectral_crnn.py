#!/usr/bin/env python3
"""
Spectral-branch CRNN with attention pooling for SER.
"""

from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, dropout2d: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout2d and dropout2d > 0:
            layers.append(nn.Dropout2d(dropout2d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _FeatureBranch(nn.Module):
    """2D CNN branch -> time sequence [B, T', D_branch]."""
    def __init__(self, input_freq_dim: int, branch_out_dim: int = 128, cnn_dropout: float = 0.2):
        super().__init__()
        self.input_freq_dim = int(input_freq_dim)
        self.branch_out_dim = int(branch_out_dim)

        self.cnn = nn.Sequential(
            _ConvBlock(1, 32, dropout2d=cnn_dropout),
            _ConvBlock(32, 64, dropout2d=cnn_dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _ConvBlock(64, 128, dropout2d=cnn_dropout),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Infer flattened feature size after conv/pool along frequency
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_freq_dim, 128)
            y = self.cnn(dummy)                         # [1, C, F', T']
            _, C, Fp, _ = y.shape
            self._flat_dim = C * Fp

        self.proj = nn.Sequential(nn.Linear(self._flat_dim, self.branch_out_dim), nn.GELU())
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(x)                                 # [B, C, F', T']
        B, C, Fp, Tp = y.shape
        y = y.permute(0, 3, 1, 2).contiguous().view(B, Tp, C * Fp)  # [B, T', C·F']
        return self.proj(y)                             # [B, T', D_branch]


class _AttentionPooling(nn.Module):
    """Learned softmax attention over time: [B, T, D] -> [B, D]."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.attn(x).squeeze(-1)                   # [B, T]
        a = torch.softmax(w, dim=1)                    # [B, T]
        return torch.sum(x * a.unsqueeze(-1), dim=1)   # [B, D]


class SpectralCRNN(nn.Module):
    def __init__(
        self,
        feature_shapes: Dict[str, int],
        num_classes: int,
        branch_out_dim: int = 128,
        rnn_hidden_size: int = 128,
        rnn_layers: int = 1,
        cnn_dropout: float = 0.2,
        fusion_dropout: float = 0.2,
        dropout: float = 0.4,
    ):
        super().__init__()
        if not feature_shapes:
            raise ValueError("feature_shapes must be a non-empty dict of {name: F_i}.")

        self.feature_names: List[str] = list(feature_shapes.keys())
        self.feature_shapes = {k: int(v) for k, v in feature_shapes.items()}

        # Per-feature CNN -> sequence branches
        self.branches = nn.ModuleList([
            _FeatureBranch(self.feature_shapes[name], branch_out_dim, cnn_dropout)
            for name in self.feature_names
        ])

        # Fusion + temporal encoder
        self.fused_dim = len(self.branches) * branch_out_dim
        self.fusion_proj = nn.Linear(self.fused_dim, self.fused_dim // 2)
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.fusion_norm = nn.LayerNorm(self.fused_dim // 2)

        self.gru = nn.GRU(
            input_size=self.fused_dim // 2,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Pooling + classifier
        self.attn_pool = _AttentionPooling(2 * rnn_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * rnn_hidden_size, num_classes)

        self._init_linear(self.fusion_proj)
        self._init_linear(self.classifier)

    @staticmethod
    def _init_linear(layer: nn.Linear) -> None:
        nn.init.trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, F_max, T]  -> logits [B, C]
        """
        if x.dtype != torch.float32: # Ensure stable dtype on MPS/with AMP
            x = x.float()

        # Per-branch CNNs -> sequences
        seqs = []
        for i, branch in enumerate(self.branches):
            name = self.feature_names[i]
            F_i = self.feature_shapes[name]
            xi = x[:, i:i+1, :F_i, :]                   # [B, 1, F_i, T]
            yi = branch(xi)                             # [B, T', D_branch]
            seqs.append(yi)

        # Fuse branches along feature dim
        fused = torch.cat(seqs, dim=2)                  # [B, T', N*D]
        fused = self.fusion_dropout(self.fusion_proj(fused))
        fused = self.fusion_norm(fused)

        # Temporal encoder + attention pooling
        rnn_out, _ = self.gru(fused)                    # [B, T', 2H]
        pooled = self.attn_pool(rnn_out)                # [B, 2H]
        return self.classifier(self.dropout(pooled))    # [B, C]

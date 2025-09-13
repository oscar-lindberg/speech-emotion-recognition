#!/usr/bin/env python3
"""
Spectral + SSL Mixture-of-Experts (MoE) head for SER.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Pooling
# ──────────────────────────────────────────────────────────────────────────────

class AttentionPool1D(nn.Module):
    """Additive attention over time: [B,T,D] -> (pooled [B,D], attn [B,T])."""
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor):
        w = self.score(x).squeeze(-1)                 # [B, T]
        a = torch.softmax(w, dim=1)                   # [B, T]
        pooled = torch.sum(x * a.unsqueeze(-1), dim=1)
        return pooled, a


class AttentiveStatsPool(nn.Module):
    """Mean + std under shared attention."""
    def __init__(self, dim: int, hidden: int = 128, eps: float = 1e-5):
        super().__init__()
        self.attn = AttentionPool1D(dim, hidden)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor):
        mean, a = self.attn(x)                                # [B, D], [B, T]
        ex2 = torch.sum((x * x) * a.unsqueeze(-1), dim=1)     # E[x^2] under attn
        var = torch.clamp(ex2 - mean * mean, min=0.0)
        std = torch.sqrt(var + self.eps)
        return torch.cat([mean, std], dim=-1), a      # [B, 2D], [B, T]


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding (sin/cos) for tiny Transformers
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Sin/cos positional encoding added to [B,T,D]."""
    def __init__(self, dim: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class SpectralSSLMoE(nn.Module):
    def __init__(
        self,
        feature_shapes: Dict[str, int],
        num_classes: int,
        *,
        # SSL
        ssl_proj_dim: Optional[int] = 256,
        ssl_encoder_layers: int = 1,
        ssl_encoder_nhead: int = 4,
        ssl_encoder_ff: int = 512,
        ssl_encoder_dropout: float = 0.10,
        # Spectral
        spec_hidden: int = 128,
        spec_conv_layers: int = 2,
        spec_kernel: int = 5,
        spec_encoder_layers: int = 0,
        spec_encoder_nhead: int = 4,
        spec_encoder_ff: int = 512,
        spec_encoder_dropout: float = 0.10,
        # Shared
        attn_hidden: int = 128,
        fusion_dim: int = 256,
        mlp_hidden: int = 256,
        gate_hidden: int = 128,
        gate_temperature: float = 1.0,
        gate_dropout: float = 0.10,
        dropout: float = 0.4,
        # Aux
        use_aux: bool = False,
    ):
        super().__init__()
        if "ssl" not in feature_shapes:
            raise ValueError("SpectralSSLMoE requires 'ssl' in feature_shapes.")
        if len(feature_shapes) < 2:
            raise ValueError("SpectralSSLMoE needs at least one spectral feature in addition to 'ssl'.")
        if int(spec_conv_layers) < 1:
            raise ValueError("spec_conv_layers must be >= 1 so that dims match spec_hidden consistently.")

        # Preserve order to match dataset stacking
        self.feature_names: List[str] = list(feature_shapes.keys())
        self.feature_shapes = {k: int(v) for k, v in feature_shapes.items()}
        self.spectral_names: List[str] = [n for n in self.feature_names if n != "ssl"]

        self.num_classes = int(num_classes)
        self.fusion_dim = int(fusion_dim)
        self.dropout_p = float(dropout)
        self.gate_temperature = float(gate_temperature)

        # SSL expert
        self.ssl_dim_in = int(self.feature_shapes["ssl"])
        self.ssl_use_proj = ssl_proj_dim is not None
        self.ssl_embed_dim = int(ssl_proj_dim) if self.ssl_use_proj else self.ssl_dim_in

        if self.ssl_use_proj:
            self.ssl_proj = nn.Sequential(
                nn.Linear(self.ssl_dim_in, self.ssl_embed_dim),
                nn.LayerNorm(self.ssl_embed_dim),
                nn.GELU(),
            )
        else:
            self.ssl_proj = nn.Identity()

        if int(ssl_encoder_layers) > 0:
            self.ssl_posenc = SinusoidalPositionalEncoding(self.ssl_embed_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.ssl_embed_dim,
                nhead=int(ssl_encoder_nhead),
                dim_feedforward=int(ssl_encoder_ff),
                dropout=float(ssl_encoder_dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.ssl_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(ssl_encoder_layers),
                                                     enable_nested_tensor=False)
        else:
            self.ssl_posenc = nn.Identity()
            self.ssl_encoder = nn.Identity()

        self.ssl_pool = AttentiveStatsPool(self.ssl_embed_dim, hidden=int(attn_hidden))
        self.ssl_post = nn.LayerNorm(2 * self.ssl_embed_dim)
        self.ssl_to_fuse = nn.Sequential(nn.Linear(2 * self.ssl_embed_dim, self.fusion_dim), nn.GELU())

        # Spectral experts
        self.spec_hidden = int(spec_hidden)
        self.spec_blocks = nn.ModuleDict()
        for name in self.spectral_names:
            F_in = int(self.feature_shapes[name])

            convs: List[nn.Module] = []
            in_ch = F_in
            for _ in range(int(spec_conv_layers)):
                convs += [
                    nn.Conv1d(in_ch, self.spec_hidden, kernel_size=int(spec_kernel),
                              stride=2, padding=int(spec_kernel // 2), bias=False),
                    nn.GELU(),
                ]
                in_ch = self.spec_hidden
            spec_conv = nn.Sequential(*convs)

            if int(spec_encoder_layers) > 0:
                spec_posenc = SinusoidalPositionalEncoding(self.spec_hidden)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=self.spec_hidden,
                    nhead=int(spec_encoder_nhead),
                    dim_feedforward=int(spec_encoder_ff),
                    dropout=float(spec_encoder_dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                spec_encoder = nn.TransformerEncoder(enc_layer, num_layers=int(spec_encoder_layers),
                                                     enable_nested_tensor=False)
            else:
                spec_posenc = nn.Identity()
                spec_encoder = nn.Identity()

            block = nn.ModuleDict({
                "conv": spec_conv,                         # [B, F_in, T] -> [B, H, T']
                "ln": nn.LayerNorm(self.spec_hidden),      # on [B, T', H]
                "pos": spec_posenc,
                "enc": spec_encoder,
                "pool": AttentiveStatsPool(self.spec_hidden, hidden=int(attn_hidden)),
                "post": nn.LayerNorm(2 * self.spec_hidden),
                "to_fuse": nn.Sequential(nn.Linear(2 * self.spec_hidden, self.fusion_dim), nn.GELU()),
            })
            self.spec_blocks[name] = block

        # Gate + classifier
        self.num_experts = 1 + len(self.spectral_names)
        self.gate = nn.Sequential(
            nn.Linear(self.num_experts * self.fusion_dim, int(gate_hidden)),
            nn.GELU(),
            nn.Dropout(float(gate_dropout)),
            nn.Linear(int(gate_hidden), self.num_experts),
        )

        self.fusion_norm = nn.LayerNorm(self.fusion_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(int(mlp_hidden), self.num_classes),
        )

        # Optional auxiliary classifier
        self.use_aux = bool(use_aux)
        self.aux_target_name: Optional[str] = "mel" if ("mel" in self.spectral_names) else (
            self.spectral_names[0] if self.spectral_names else None
        )
        if self.use_aux:
            self.aux_classifier = nn.Linear(self.fusion_dim, self.num_classes)

        self._init_weights()

    # ── utils ────────────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def _idx_and_F(self, name: str) -> Tuple[int, int]:
        i = self.feature_names.index(name)
        F_i = self.feature_shapes[name]
        return i, F_i


    def forward(self, x: torch.Tensor, return_extras: bool = False):
        """
        x: [B, N, F_max, T]
        Returns logits [B, C] or (logits, extras) if return_extras=True.
        """
        if x.dtype != torch.float32:  # MPS/LayerNorm/Transformer prefer fp32
            x = x.float()

        # SSL expert
        ssl_idx, F_ssl = self._idx_and_F("ssl")
        ssl = x[:, ssl_idx:ssl_idx + 1, :F_ssl, :].squeeze(1)     # [B, F_ssl, T]
        ssl = ssl.transpose(1, 2).contiguous()        # [B, T, F_ssl]
        ssl = self.ssl_proj(ssl)
        ssl = self.ssl_posenc(ssl)
        ssl = self.ssl_encoder(ssl)                               # [B, T, Dssl]
        ssl_vec, ssl_attn = self.ssl_pool(ssl)                    # [B, 2*Dssl], [B, T]
        ssl_vec = self.ssl_post(ssl_vec)
        ssl_fuse = self.ssl_to_fuse(ssl_vec)                      # [B, fusion_dim]

        # Spectral experts
        spec_fuses: List[torch.Tensor] = []
        spec_attn: Dict[str, torch.Tensor] = {}
        for name in self.spectral_names:
            F_in = self.feature_shapes[name]
            idx = self.feature_names.index(name)
            xi = x[:, idx:idx + 1, :F_in, :].squeeze(1)          # [B, F_in, T]
            blk = self.spec_blocks[name]

            h = blk["conv"](xi)                                  # [B, H, T']
            h = h.transpose(1, 2).contiguous()                   # [B, T', H]
            h = blk["ln"](h)
            h = blk["pos"](h)
            h = blk["enc"](h)
            v, a = blk["pool"](h)                                # [B, 2H], [B, T']
            v = blk["post"](v)
            v = blk["to_fuse"](v)                                # [B, fusion_dim]
            spec_fuses.append(v)
            spec_attn[name] = a

        # Gating and fusion
        all_fuses = [ssl_fuse] + spec_fuses                       # K vectors [B, fusion_dim]
        gate_in = torch.cat(all_fuses, dim=-1)                    # [B, K*fusion_dim]
        gate_logits = self.gate(gate_in)                          # [B, K]
        if self.gate_temperature != 1.0:
            gate_logits = gate_logits / self.gate_temperature
        gate_probs = torch.softmax(gate_logits, dim=-1)           # [B, K]

        stacked = torch.stack(all_fuses, dim=1)                   # [B, K, fusion_dim]
        fused = torch.sum(gate_probs.unsqueeze(-1) * stacked, dim=1)  # [B, fusion_dim]
        fused = self.fusion_norm(fused)                           # stabilize
        logits = self.classifier(self.dropout(fused))             # [B, C]

        if not return_extras and not self.use_aux:
            return logits

        extras = {
            "experts": ["ssl"] + self.spectral_names,
            "gate_probs": gate_probs,
            "attn": {"ssl": ssl_attn, **spec_attn},
        }

        if self.use_aux:
            # Choose configured spectral target if available; fall back to mean of spectral
            if hasattr(self, "aux_classifier"):
                if self.aux_target_name in self.spectral_names:
                    k = 1 + self.spectral_names.index(self.aux_target_name)
                    aux_vec = stacked[:, k, :]
                else:
                    aux_vec = stacked[:, 1:, :].mean(dim=1) if stacked.size(1) > 1 else stacked[:, 0, :]
                extras["aux_logits"] = self.aux_classifier(aux_vec)

        return logits, extras

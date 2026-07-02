from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.models.CrossCompartment.cross_compartment import DeepEnhancedBranch, Mlp, SSScanDNAHybridModel, TokenBridge, ensure_finite, reverse_complement


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def copy(self):
        return AttrDict(super().copy())


def make_transformer_cfg(d_model: int, block_size: int, num_heads: int) -> AttrDict:
    return AttrDict(
        {
            "hidden_size": d_model,
            "norm_eps": 1e-5,
            "max_position_embeddings": max(block_size, 2048),
            "hidden_ratio": 4.0,
            "hidden_act": "swish",
            "fuse_swiglu": True,
            "attn": {
            "num_heads": num_heads,
            "num_kv_heads": num_heads,
            "qkv_bias": False,
            "window_size": min(block_size, 512),
            "rope_theta": 10000,
            },
        }
    )


def make_comba_cfg(d_model: int, num_heads: int) -> dict:
    return {
        "hidden_size": d_model,
        "expand_v": 1,
        "head_dim": max(16, d_model // max(1, num_heads)),
        "num_heads": num_heads,
        "use_gate": True,
        "mode": "chunk",
        "use_short_conv": True,
        "correction_factor": 0.02,
        "conv_size": 4,
        "norm_eps": 1e-5,
    }


class MeanPoolHead(nn.Module):
    def __init__(self, d_model: int, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.out(self.dropout(self.norm(pooled)))


class CrossStrandTrackEncoder(nn.Module):
    """CrossCompartment-style dual-strand encoder for provided plus/minus signal tracks."""

    def __init__(
        self,
        d_model: int = 256,
        block_size: int = 2048,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        transformer_cfg = make_transformer_cfg(d_model, block_size, num_heads)
        comba_cfg = make_comba_cfg(d_model, num_heads)
        self.plus_embed = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.minus_embed = nn.Conv1d(1, d_model, kernel_size=9, padding=4)
        self.plus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.minus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.bridge = TokenBridge(d_model, dropout=dropout)
        self.proj_plus = Mlp(d_model, d_model * 2, d_model, activation=F.gelu, return_residual=True)
        self.proj_minus = Mlp(d_model, d_model * 2, d_model, activation=F.gelu, return_residual=True)
        self.gate_fuse = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        plus = tracks[..., 0:1].transpose(1, 2)
        minus = tracks[..., 1:2].transpose(1, 2)
        plus = self.dropout(F.gelu(self.plus_embed(plus))).transpose(1, 2)
        minus = self.dropout(F.gelu(self.minus_embed(minus))).transpose(1, 2)

        plus = self.plus_core(plus)
        minus = self.minus_core(minus)
        plus, minus = self.bridge(plus, minus)

        plus_delta, plus_resid = self.proj_plus(plus)
        minus_delta, minus_resid = self.proj_minus(minus)
        plus = plus_delta + plus_resid
        minus = minus_delta + minus_resid

        gate = torch.sigmoid(self.gate_fuse(torch.cat([plus, minus], dim=-1)))
        fused = F.layer_norm(gate * plus + (1 - gate) * minus, (plus.size(-1),))
        return ensure_finite(fused, "cross_strand_track_encoder_fused")


class CrossCompartmentDNAABClassifier(nn.Module):
    """AB classifier that reuses CrossCompartment's DNA token backbone."""

    def __init__(
        self,
        d_model: int = 256,
        block_size: int = 2048,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.backbone = SSScanDNAHybridModel(
            alphabet_size=5,
            d_model=d_model,
            block_size=block_size,
            comba_cfg=make_comba_cfg(d_model, num_heads),
            transformer_cfg=make_transformer_cfg(d_model, block_size, num_heads),
            depth=depth,
            drop_path_rates=[0.0] * depth,
            pretrain=False,
            for_representation=True,
            use_ema_teacher=False,
            use_final_conv=False,
            use_rc_kl=False,
            use_barlow=False,
            use_tv=False,
            gate_freeze_steps=0,
            dropout=dropout,
        )
        self.head = MeanPoolHead(d_model, n_classes=n_classes, dropout=dropout)

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        features, _ = self.backbone(seq_ids)
        return self.head(features)


class CrossCompartmentRnaStrandABClassifier(nn.Module):
    """AB classifier for plus/minus RO-seq or PRO-seq signal tracks."""

    def __init__(
        self,
        d_model: int = 256,
        block_size: int = 2048,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.encoder = CrossStrandTrackEncoder(
            d_model=d_model,
            block_size=block_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.head = MeanPoolHead(d_model, n_classes=n_classes, dropout=dropout)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(tracks))


class CrossCompartmentFusionCompartmentPredictor(nn.Module):
    """Fuse DNA sequence and plus/minus RO-seq features to predict compartment score."""

    def __init__(
        self,
        d_model: int = 256,
        block_size: int = 4096,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        output: str = "regression",
    ):
        super().__init__()
        if output not in {"regression", "binary"}:
            raise ValueError("output must be 'regression' or 'binary'")
        self.output = output
        self.dna = CrossCompartmentDNAABClassifier(
            d_model=d_model,
            block_size=block_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            n_classes=2,
        ).backbone
        self.rna = CrossStrandTrackEncoder(
            d_model=d_model,
            block_size=block_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.modality_gate = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1 if output == "regression" else 2),
        )

    def forward(self, seq_ids: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        dna_features, _ = self.dna(seq_ids)
        rna_features = self.rna(tracks)
        gate_input = torch.cat(
            [dna_features, rna_features, dna_features - rna_features, dna_features * rna_features],
            dim=-1,
        )
        gate = torch.sigmoid(self.modality_gate(gate_input))
        fused = self.norm(gate * dna_features + (1 - gate) * rna_features)
        pooled = fused.mean(dim=1)
        out = self.head(self.dropout(pooled))
        return out.squeeze(-1) if self.output == "regression" else out


class CrossCompartmentFusionRangePredictor(nn.Module):
    """Predict a vector of compartment scores for all bins in a larger genomic range."""

    def __init__(
        self,
        d_model: int = 128,
        block_size: int = 1024,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bin_model = CrossCompartmentFusionCompartmentPredictor(
            d_model=d_model,
            block_size=block_size,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
            output="regression",
        )

    def forward(self, seq_ids: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        batch, n_bins, seq_len = seq_ids.shape
        _, _, track_len, channels = tracks.shape
        pred = self.bin_model(
            seq_ids.reshape(batch * n_bins, seq_len),
            tracks.reshape(batch * n_bins, track_len, channels),
        )
        return pred.reshape(batch, n_bins)


def normalize_stem_kernels(strides: tuple[int, ...], kernels: tuple[int, ...] | None) -> tuple[int, ...]:
    if kernels is None:
        return tuple(15 for _ in strides)
    if len(kernels) != len(strides):
        raise ValueError(f"bp_stem_kernels length {len(kernels)} must match strides length {len(strides)}")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in kernels):
        raise ValueError("All bp stem kernels must be positive odd integers")
    return tuple(int(kernel) for kernel in kernels)


class StridedConvStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        layers = []
        channels = in_channels
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            layers.extend(
                [
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=stride, padding=kernel // 2),
                    nn.GroupNorm(8 if d_model % 8 == 0 else 1, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            channels = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2).contiguous()


class ConvPoolStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        layers = []
        channels = in_channels
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            layers.extend(
                [
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(8 if d_model % 8 == 0 else 1, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                ]
            )
            channels = d_model
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).transpose(1, 2).contiguous()


class CrossStrandConvPoolStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        use_bridge: bool = True,
        align_minus_by_flip: bool = False,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.align_minus_by_flip = align_minus_by_flip
        self.plus_blocks = nn.ModuleList()
        self.minus_blocks = nn.ModuleList()
        self.bridges = nn.ModuleList()
        channels = in_channels
        groups = 8 if d_model % 8 == 0 else 1
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            self.plus_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                )
            )
            self.minus_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                )
            )
            self.bridges.append(TokenBridge(d_model, dropout=dropout) if use_bridge else nn.Identity())
            channels = d_model

    def forward(self, plus: torch.Tensor, minus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for plus_block, minus_block, bridge in zip(self.plus_blocks, self.minus_blocks, self.bridges):
            plus = plus_block(plus)
            minus = minus_block(minus)
            plus_tok = plus.transpose(1, 2).contiguous()
            minus_tok = minus.transpose(1, 2).contiguous()
            if isinstance(bridge, TokenBridge):
                if self.align_minus_by_flip:
                    minus_bridge = torch.flip(minus_tok, dims=[1])
                    plus_tok, minus_bridge = bridge(plus_tok, minus_bridge)
                    minus_tok = torch.flip(minus_bridge, dims=[1])
                else:
                    plus_tok, minus_tok = bridge(plus_tok, minus_tok)
            plus = plus_tok.transpose(1, 2).contiguous()
            minus = minus_tok.transpose(1, 2).contiguous()
        return plus.transpose(1, 2).contiguous(), minus.transpose(1, 2).contiguous()


class LiteCrossStrandConvPoolStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        align_minus_by_flip: bool = False,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.align_minus_by_flip = align_minus_by_flip
        self.plus_blocks = nn.ModuleList()
        self.minus_blocks = nn.ModuleList()
        self.proj_plus_to_minus = nn.ModuleList()
        self.proj_minus_to_plus = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norm_plus = nn.ModuleList()
        self.norm_minus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        channels = in_channels
        groups = 8 if d_model % 8 == 0 else 1
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            self.plus_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                )
            )
            self.minus_blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                )
            )
            self.proj_plus_to_minus.append(nn.Linear(d_model, d_model))
            self.proj_minus_to_plus.append(nn.Linear(d_model, d_model))
            self.gates.append(nn.Linear(d_model * 4, d_model * 2))
            self.norm_plus.append(nn.LayerNorm(d_model))
            self.norm_minus.append(nn.LayerNorm(d_model))
            channels = d_model

    def _interact(self, plus_tok: torch.Tensor, minus_tok: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([plus_tok, minus_tok, plus_tok - minus_tok, plus_tok * minus_tok], dim=-1)
        gate_plus, gate_minus = self.gates[idx](z).chunk(2, dim=-1)
        gate_plus = torch.sigmoid(gate_plus)
        gate_minus = torch.sigmoid(gate_minus)
        plus_update = self.proj_minus_to_plus[idx](minus_tok)
        minus_update = self.proj_plus_to_minus[idx](plus_tok)
        plus_tok = self.norm_plus[idx](plus_tok + self.dropout(gate_plus * plus_update))
        minus_tok = self.norm_minus[idx](minus_tok + self.dropout(gate_minus * minus_update))
        return plus_tok, minus_tok

    def forward(self, plus: torch.Tensor, minus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for idx, (plus_block, minus_block) in enumerate(zip(self.plus_blocks, self.minus_blocks)):
            plus = plus_block(plus)
            minus = minus_block(minus)
            plus_tok = plus.transpose(1, 2).contiguous()
            minus_tok = minus.transpose(1, 2).contiguous()
            if self.align_minus_by_flip:
                minus_tok = torch.flip(minus_tok, dims=[1])
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
                minus_tok = torch.flip(minus_tok, dims=[1])
            else:
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
            plus = plus_tok.transpose(1, 2).contiguous()
            minus = minus_tok.transpose(1, 2).contiguous()
        return plus.transpose(1, 2).contiguous(), minus.transpose(1, 2).contiguous()


class PoolFirstLiteCrossStrandStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        align_minus_by_flip: bool = False,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.align_minus_by_flip = align_minus_by_flip
        self.plus_blocks = nn.ModuleList()
        self.minus_blocks = nn.ModuleList()
        self.proj_plus_to_minus = nn.ModuleList()
        self.proj_minus_to_plus = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norm_plus = nn.ModuleList()
        self.norm_minus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        channels = in_channels
        groups = 8 if d_model % 8 == 0 else 1
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            self.plus_blocks.append(
                nn.Sequential(
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.minus_blocks.append(
                nn.Sequential(
                    nn.AvgPool1d(kernel_size=stride, stride=stride),
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.proj_plus_to_minus.append(nn.Linear(d_model, d_model))
            self.proj_minus_to_plus.append(nn.Linear(d_model, d_model))
            self.gates.append(nn.Linear(d_model * 4, d_model * 2))
            self.norm_plus.append(nn.LayerNorm(d_model))
            self.norm_minus.append(nn.LayerNorm(d_model))
            channels = d_model

    def _interact(self, plus_tok: torch.Tensor, minus_tok: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([plus_tok, minus_tok, plus_tok - minus_tok, plus_tok * minus_tok], dim=-1)
        gate_plus, gate_minus = self.gates[idx](z).chunk(2, dim=-1)
        plus_update = self.proj_minus_to_plus[idx](minus_tok)
        minus_update = self.proj_plus_to_minus[idx](plus_tok)
        plus_tok = self.norm_plus[idx](plus_tok + self.dropout(torch.sigmoid(gate_plus) * plus_update))
        minus_tok = self.norm_minus[idx](minus_tok + self.dropout(torch.sigmoid(gate_minus) * minus_update))
        return plus_tok, minus_tok

    def forward(self, plus: torch.Tensor, minus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for idx, (plus_block, minus_block) in enumerate(zip(self.plus_blocks, self.minus_blocks)):
            plus = plus_block(plus)
            minus = minus_block(minus)
            plus_tok = plus.transpose(1, 2).contiguous()
            minus_tok = minus.transpose(1, 2).contiguous()
            if self.align_minus_by_flip:
                minus_tok = torch.flip(minus_tok, dims=[1])
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
                minus_tok = torch.flip(minus_tok, dims=[1])
            else:
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
            plus = plus_tok.transpose(1, 2).contiguous()
            minus = minus_tok.transpose(1, 2).contiguous()
        return plus.transpose(1, 2).contiguous(), minus.transpose(1, 2).contiguous()


class ConvFirstLiteCrossStrandStem1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        strides=(10, 10, 10, 10, 5),
        dropout: float = 0.1,
        align_minus_by_flip: bool = False,
        kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.align_minus_by_flip = align_minus_by_flip
        self.plus_convs = nn.ModuleList()
        self.minus_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.proj_plus_to_minus = nn.ModuleList()
        self.proj_minus_to_plus = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norm_plus = nn.ModuleList()
        self.norm_minus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        channels = in_channels
        groups = 8 if d_model % 8 == 0 else 1
        kernels = normalize_stem_kernels(tuple(strides), kernels)
        for stride, kernel in zip(strides, kernels):
            self.plus_convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.minus_convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, d_model, kernel_size=kernel, stride=1, padding=kernel // 2),
                    nn.GroupNorm(groups, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            self.pools.append(nn.AvgPool1d(kernel_size=stride, stride=stride))
            self.proj_plus_to_minus.append(nn.Linear(d_model, d_model))
            self.proj_minus_to_plus.append(nn.Linear(d_model, d_model))
            self.gates.append(nn.Linear(d_model * 4, d_model * 2))
            self.norm_plus.append(nn.LayerNorm(d_model))
            self.norm_minus.append(nn.LayerNorm(d_model))
            channels = d_model

    def _interact(self, plus_tok: torch.Tensor, minus_tok: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([plus_tok, minus_tok, plus_tok - minus_tok, plus_tok * minus_tok], dim=-1)
        gate_plus, gate_minus = self.gates[idx](z).chunk(2, dim=-1)
        plus_update = self.proj_minus_to_plus[idx](minus_tok)
        minus_update = self.proj_plus_to_minus[idx](plus_tok)
        plus_tok = self.norm_plus[idx](plus_tok + self.dropout(torch.sigmoid(gate_plus) * plus_update))
        minus_tok = self.norm_minus[idx](minus_tok + self.dropout(torch.sigmoid(gate_minus) * minus_update))
        return plus_tok, minus_tok

    def forward(self, plus: torch.Tensor, minus: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for idx, (plus_conv, minus_conv, pool) in enumerate(zip(self.plus_convs, self.minus_convs, self.pools)):
            plus = plus_conv(plus)
            minus = minus_conv(minus)
            plus_tok = plus.transpose(1, 2).contiguous()
            minus_tok = minus.transpose(1, 2).contiguous()
            if self.align_minus_by_flip:
                minus_tok = torch.flip(minus_tok, dims=[1])
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
                minus_tok = torch.flip(minus_tok, dims=[1])
            else:
                plus_tok, minus_tok = self._interact(plus_tok, minus_tok, idx)
            plus = pool(plus_tok.transpose(1, 2).contiguous())
            minus = pool(minus_tok.transpose(1, 2).contiguous())
        return plus.transpose(1, 2).contiguous(), minus.transpose(1, 2).contiguous()


def make_bp_stem(
    stem_type: str,
    in_channels: int,
    d_model: int,
    strides: tuple[int, ...],
    dropout: float,
    kernels: tuple[int, ...] | None = None,
) -> nn.Module:
    if stem_type == "strided_conv":
        return StridedConvStem1D(in_channels, d_model, strides=strides, dropout=dropout, kernels=kernels)
    if stem_type == "conv_pool":
        return ConvPoolStem1D(in_channels, d_model, strides=strides, dropout=dropout, kernels=kernels)
    raise ValueError(f"Unknown bp stem type: {stem_type}")


class BPLevelCrossStrandDNAEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        n_bins: int,
        dropout: float = 0.1,
        use_strand_bridge: bool = True,
        stem_strides: tuple[int, ...] = (10, 10, 10, 10, 5),
        stem_type: str = "strided_conv",
        stem_kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.use_strand_bridge = use_strand_bridge
        self.stem_type = stem_type
        transformer_cfg = make_transformer_cfg(d_model, n_bins, num_heads)
        comba_cfg = make_comba_cfg(d_model, num_heads)
        if stem_type == "cross_conv_pool":
            self.cross_stem = CrossStrandConvPoolStem1D(
                5,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                use_bridge=use_strand_bridge,
                align_minus_by_flip=True,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_conv_pool_lite":
            self.cross_stem = LiteCrossStrandConvPoolStem1D(
                5,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=True,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_poolfirst_lite":
            self.cross_stem = PoolFirstLiteCrossStrandStem1D(
                5,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=True,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_convfirst_lite":
            self.cross_stem = ConvFirstLiteCrossStrandStem1D(
                5,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=True,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        else:
            self.cross_stem = None
            self.plus_stem = make_bp_stem(stem_type, 5, d_model, stem_strides, dropout, kernels=stem_kernels)
            self.minus_stem = make_bp_stem(stem_type, 5, d_model, stem_strides, dropout, kernels=stem_kernels)
        self.plus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.minus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.bridge = TokenBridge(d_model, dropout=dropout) if use_strand_bridge else None
        self.gate_fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, seq_ids: torch.Tensor) -> torch.Tensor:
        plus = F.one_hot(seq_ids, num_classes=5).float().transpose(1, 2)
        minus_ids = reverse_complement(seq_ids)
        minus = F.one_hot(minus_ids, num_classes=5).float().transpose(1, 2)
        if self.cross_stem is not None:
            plus, minus = self.cross_stem(plus, minus)
        else:
            plus = self.plus_stem(plus)
            minus = self.minus_stem(minus)
        plus = self.plus_core(plus)
        minus = self.minus_core(minus)
        minus = torch.flip(minus, dims=[1])
        if self.bridge is not None:
            plus, minus = self.bridge(plus, minus)
        gate = torch.sigmoid(self.gate_fuse(torch.cat([plus, minus], dim=-1)))
        return F.layer_norm(gate * plus + (1 - gate) * minus, (plus.size(-1),))


class BPLevelCrossStrandTrackEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        depth: int,
        num_heads: int,
        n_bins: int,
        dropout: float = 0.1,
        use_strand_bridge: bool = True,
        stem_strides: tuple[int, ...] = (10, 10, 10, 10, 5),
        stem_type: str = "strided_conv",
        stem_kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.use_strand_bridge = use_strand_bridge
        self.stem_type = stem_type
        transformer_cfg = make_transformer_cfg(d_model, n_bins, num_heads)
        comba_cfg = make_comba_cfg(d_model, num_heads)
        if stem_type == "cross_conv_pool":
            self.cross_stem = CrossStrandConvPoolStem1D(
                1,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                use_bridge=use_strand_bridge,
                align_minus_by_flip=False,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_conv_pool_lite":
            self.cross_stem = LiteCrossStrandConvPoolStem1D(
                1,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=False,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_poolfirst_lite":
            self.cross_stem = PoolFirstLiteCrossStrandStem1D(
                1,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=False,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        elif stem_type == "cross_convfirst_lite":
            self.cross_stem = ConvFirstLiteCrossStrandStem1D(
                1,
                d_model,
                strides=stem_strides,
                dropout=dropout,
                align_minus_by_flip=False,
                kernels=stem_kernels,
            )
            self.plus_stem = None
            self.minus_stem = None
        else:
            self.cross_stem = None
            self.plus_stem = make_bp_stem(stem_type, 1, d_model, stem_strides, dropout, kernels=stem_kernels)
            self.minus_stem = make_bp_stem(stem_type, 1, d_model, stem_strides, dropout, kernels=stem_kernels)
        self.plus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.minus_core = DeepEnhancedBranch(d_model, comba_cfg, transformer_cfg, depth=depth, drop_path_rates=[0.0] * depth)
        self.bridge = TokenBridge(d_model, dropout=dropout) if use_strand_bridge else None
        self.gate_fuse = nn.Linear(2 * d_model, d_model)

    def forward(self, tracks: torch.Tensor) -> torch.Tensor:
        plus = tracks[..., 0:1].transpose(1, 2)
        minus = tracks[..., 1:2].transpose(1, 2)
        if self.cross_stem is not None:
            plus, minus = self.cross_stem(plus, minus)
        else:
            plus = self.plus_stem(plus)
            minus = self.minus_stem(minus)
        plus = self.plus_core(plus)
        minus = self.minus_core(minus)
        if self.bridge is not None:
            plus, minus = self.bridge(plus, minus)
        gate = torch.sigmoid(self.gate_fuse(torch.cat([plus, minus], dim=-1)))
        return F.layer_norm(gate * plus + (1 - gate) * minus, (plus.size(-1),))


class ResidualConvBlock1D(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for same-length residual convolution")
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.pointwise(y).transpose(1, 2)
        return residual + self.dropout(y)


class ResidualConvDecoder1D(nn.Module):
    def __init__(self, d_model: int, layers: int = 0, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.enabled = layers > 0
        self.layers = nn.Sequential(
            *[ResidualConvBlock1D(d_model, kernel_size=kernel_size, dropout=dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(d_model) if self.enabled else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.layers(x))


class TransformerTokenDecoder(nn.Module):
    def __init__(self, d_model: int, layers: int = 0, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.enabled = layers > 0
        if self.enabled:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.layers = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.layers(x))


class BPLevelFusionRangePredictor(nn.Module):
    """BP-level DNA + RO-seq model with configurable strided-conv binning."""

    FUSION_METHODS = {"gate", "concat", "simple_concat", "channel_concat", "cross_attention"}

    def __init__(
        self,
        d_model: int = 128,
        n_bins: int = 40,
        depth: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        ablation: str = "full",
        fusion_method: str = "gate",
        decoder_layers: int = 0,
        decoder_kernel_size: int = 5,
        decoder_type: str = "conv",
        encoder_chunk_bins: int | None = None,
        encoder_checkpoint: bool = True,
        bp_downsample_strides: tuple[int, ...] = (10, 10, 10, 10, 5),
        bp_stem_type: str = "strided_conv",
        bp_stem_kernels: tuple[int, ...] | None = None,
    ):
        super().__init__()
        if ablation not in {"full", "dna_only", "ro_only", "no_strand_fusion", "no_modality_gate"}:
            raise ValueError(f"Unknown ablation: {ablation}")
        if fusion_method not in self.FUSION_METHODS:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
        if decoder_type not in {"conv", "transformer"}:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        if bp_stem_type not in {"strided_conv", "conv_pool", "cross_conv_pool", "cross_conv_pool_lite", "cross_poolfirst_lite", "cross_convfirst_lite"}:
            raise ValueError(f"Unknown bp_stem_type: {bp_stem_type}")
        if encoder_chunk_bins is not None and encoder_chunk_bins <= 0:
            raise ValueError("encoder_chunk_bins must be positive when set")
        if encoder_chunk_bins is not None and n_bins % encoder_chunk_bins != 0:
            raise ValueError(f"n_bins={n_bins} must be divisible by encoder_chunk_bins={encoder_chunk_bins}")
        if not bp_downsample_strides:
            raise ValueError("bp_downsample_strides must contain at least one stride")
        downsample_factor = 1
        for stride in bp_downsample_strides:
            if stride <= 0:
                raise ValueError("bp_downsample_strides must be positive")
            downsample_factor *= int(stride)
        if downsample_factor <= 0:
            raise ValueError("Invalid bp_downsample_strides")
        self.ablation = ablation
        self.fusion_method = "concat" if fusion_method == "channel_concat" else fusion_method
        self.n_bins = n_bins
        self.encoder_chunk_bins = encoder_chunk_bins
        self.encoder_checkpoint = encoder_checkpoint
        self.bp_downsample_strides = tuple(int(x) for x in bp_downsample_strides)
        self.bp_stem_type = bp_stem_type
        self.bp_stem_kernels = normalize_stem_kernels(self.bp_downsample_strides, bp_stem_kernels)
        encoder_n_bins = encoder_chunk_bins or n_bins
        use_strand_bridge = ablation != "no_strand_fusion"
        self.dna = None if ablation == "ro_only" else BPLevelCrossStrandDNAEncoder(
            d_model,
            depth,
            num_heads,
            encoder_n_bins,
            dropout,
            use_strand_bridge=use_strand_bridge,
            stem_strides=self.bp_downsample_strides,
            stem_type=self.bp_stem_type,
            stem_kernels=self.bp_stem_kernels,
        )
        self.rna = None if ablation == "dna_only" else BPLevelCrossStrandTrackEncoder(
            d_model,
            depth,
            num_heads,
            encoder_n_bins,
            dropout,
            use_strand_bridge=use_strand_bridge,
            stem_strides=self.bp_downsample_strides,
            stem_type=self.bp_stem_type,
            stem_kernels=self.bp_stem_kernels,
        )
        self.modality_gate = nn.Linear(d_model * 4, d_model)
        self.concat_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.simple_concat_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.cross_dna_from_rna = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_rna_from_dna = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_fusion = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        if decoder_type == "transformer":
            self.decoder = TransformerTokenDecoder(
                d_model=d_model,
                layers=decoder_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.decoder = ResidualConvDecoder1D(
                d_model=d_model,
                layers=decoder_layers,
                kernel_size=decoder_kernel_size,
                dropout=dropout,
            )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _concat_features(self, dna_features: torch.Tensor, rna_features: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [dna_features, rna_features, dna_features - rna_features, dna_features * rna_features],
            dim=-1,
        )

    def _simple_concat_features(self, dna_features: torch.Tensor, rna_features: torch.Tensor) -> torch.Tensor:
        return torch.cat([dna_features, rna_features], dim=-1)

    def _fuse_modalities(self, dna_features: torch.Tensor, rna_features: torch.Tensor) -> torch.Tensor:
        if self.ablation == "no_modality_gate":
            return self.norm(0.5 * (dna_features + rna_features))

        if self.fusion_method == "gate":
            gate = torch.sigmoid(self.modality_gate(self._concat_features(dna_features, rna_features)))
            return self.norm(gate * dna_features + (1 - gate) * rna_features)

        if self.fusion_method == "concat":
            fused = self.concat_fusion(self._concat_features(dna_features, rna_features))
            return self.norm(fused + 0.5 * (dna_features + rna_features))

        if self.fusion_method == "simple_concat":
            fused = self.simple_concat_fusion(self._simple_concat_features(dna_features, rna_features))
            return self.norm(fused + 0.5 * (dna_features + rna_features))

        dna_ctx, _ = self.cross_dna_from_rna(dna_features, rna_features, rna_features, need_weights=False)
        rna_ctx, _ = self.cross_rna_from_dna(rna_features, dna_features, dna_features, need_weights=False)
        fused = self.cross_fusion(torch.cat([dna_features, rna_features, dna_ctx, rna_ctx], dim=-1))
        return self.norm(fused + 0.5 * (dna_features + rna_features))

    def _encode_maybe_chunked(self, encoder: nn.Module | None, x: torch.Tensor) -> torch.Tensor | None:
        if encoder is None:
            return None
        if self.encoder_chunk_bins is None or self.encoder_chunk_bins >= self.n_bins:
            return encoder(x)

        n_chunks = self.n_bins // self.encoder_chunk_bins
        length = x.size(1)
        if length % n_chunks != 0:
            raise ValueError(f"Input length {length} is not divisible by {n_chunks} encoder chunks")
        chunk_len = length // n_chunks

        features_by_chunk = []
        use_checkpoint = self.encoder_checkpoint and self.training and torch.is_grad_enabled() and x.is_cuda
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_len
            stop = start + chunk_len
            chunk = x[:, start:stop].contiguous()
            if use_checkpoint:
                features = checkpoint(encoder, chunk, use_reentrant=False)
            else:
                features = encoder(chunk)
            if features.size(1) != self.encoder_chunk_bins:
                raise ValueError(
                    f"Encoder emitted {features.size(1)} bins per chunk; expected {self.encoder_chunk_bins}"
                )
            features_by_chunk.append(features)
        return torch.cat(features_by_chunk, dim=1)

    def forward(self, seq_ids: torch.Tensor, tracks: torch.Tensor) -> torch.Tensor:
        dna_features = self._encode_maybe_chunked(self.dna, seq_ids)
        rna_features = self._encode_maybe_chunked(self.rna, tracks)
        if dna_features is None:
            fused = self.norm(rna_features)
            return self.head(self.decoder(fused)).squeeze(-1)
        if rna_features is None:
            fused = self.norm(dna_features)
            return self.head(self.decoder(fused)).squeeze(-1)
        fused = self._fuse_modalities(dna_features, rna_features)
        return self.head(self.decoder(fused)).squeeze(-1)

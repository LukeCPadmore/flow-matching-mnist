from dataclasses import dataclass
from typing import Tuple, Literal
import torch.nn as nn
import torch.optim

ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]
OptimName = Literal["adam", "adamw", "sgd"]

def make_activation(name: ActivationName) -> type[nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "silu":
        return nn.SiLU
    if name == "gelu":
        return nn.GELU
    raise ValueError(f"Unknown activation {name}")

def make_upsample(name: UpsampleMode, channels: int):
    if name == "nearest":
        return nn.Upsample(scale_factor=2, mode="nearest")
    if name == "bilinear":
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if name == "convtranspose":
        return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
    raise ValueError(f"Unknown upsample {name}")

@dataclass(frozen=True)
class UNetConfig:
    # architecture
    channels: Tuple[int, ...]          
    d_trunk: int = 32                  
    d_concat: int = 8                 
    group_norm_size: int = 8

    # time embedding params
    d_time: int = 128
    max_time_period: float = 10000.0

    # Upsampling and activation function
    activation: ActivationName = "silu"
    upsample_mode: UpsampleMode = "nearest"


@dataclass(frozen=True)
class OptimConfig:
    name: OptimName = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4

def make_optimizer(cfg: OptimConfig, params):
    if cfg.name == "adam":
        return torch.optim.Adam(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    if cfg.name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    if cfg.name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimiser {cfg.name}")






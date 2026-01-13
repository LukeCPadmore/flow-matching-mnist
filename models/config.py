from dataclasses import dataclass
from typing import Tuple, Literal
import torch.nn as nn

ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]

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
    # --- architecture ---
    channels: Tuple[int, ...]          
    d_trunk: int = 32                  
    d_concat: int = 8                 
    group_norm_size: int = 8

    # --- time embedding ---
    d_time: int = 128
    max_time_period: float = 10000.0

    # --- choices ---
    activation: ActivationName = "silu"
    upsample_mode: UpsampleMode = "nearest"




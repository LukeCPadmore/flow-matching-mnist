from dataclasses import dataclass
from typing import Tuple, Literal
import torch.nn as nn



def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation {name}")

def make_upsample(name: str, channels: int):
    if name == "nearest":
        return nn.Upsample(scale_factor=2, mode="nearest")
    if name == "bilinear":
        return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    if name == "convtranspose":
        return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
    raise ValueError(f"Unknown upsample {name}")

@dataclass(frozen=True)
class UNetConfig:
    base_channels: int
    num_res_blocks: int
    dropout: float
    channel_mults: Tuple[int, ...]
    lr: float
    weight_decay: float

    activation: Literal["relu", "silu", "gelu"] = "silu"
    upsample: Literal["nearest", "bilinear", "convtranspose"] = "nearest"




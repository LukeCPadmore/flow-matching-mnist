from dataclasses import dataclass
from typing import Tuple, Literal
import torch.nn as nn
import torch.optim
import optuna

ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]
OptimName = Literal["adam", "adamw", "sgd"]


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

    @classmethod
    def make_activation(cls, name: ActivationName) -> type[nn.Module]:
        name = name.lower()
        if name == "relu":
            return nn.ReLU
        if name == "silu":
            return nn.SiLU
        if name == "gelu":
            return nn.GELU
        raise ValueError(f"Unknown activation {name}")

    @classmethod
    def make_upsample(cls, name: UpsampleMode, channels: int):
        if name == "nearest":
            return nn.Upsample(scale_factor=2, mode="nearest")
        if name == "bilinear":
            return nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        if name == "convtranspose":
            return nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)
        raise ValueError(f"Unknown upsample {name}")

    @classmethod
    def sample_unet_cfg(cls, trial: optuna.Trial, *, fixed=None):
        fixed = dict(fixed or {})

        def get(name, sampler):
            return fixed.get(name, sampler())
        
        


@dataclass(frozen=True)
class OptimConfig:
    name: OptimName = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4

    def make_optimizer(self, params):
        if self.name == "adam":
            return torch.optim.Adam(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        if self.name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        if self.name == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimiser {self.name}")

    @classmethod
    def sample_optim_cfg(cls, trial: optuna.Trial, *, fixed=None):
        fixed = dict(fixed or {})

        def get(name, sampler):
            return fixed.get(name, sampler())

        name = get(
            "name", lambda: trial.suggest_categorical("optim", ["adamw", "adam"])
        )
        lr = get("lr", lambda: trial.suggest_float("lr", 1e-5, 5e-4, log=True))
        wd = get(
            "weight_decay",
            lambda: trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )

        return cls(name=name, lr=float(lr), weight_decay=float(wd))

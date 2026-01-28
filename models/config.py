from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Tuple, Literal, Mapping, Any, Sequence
import torch.nn as nn
import torch.optim
import optuna

OptimName = Literal["adam", "adamw", "sgd"]
ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]


@dataclass
class SamplerCtx:
    trial: optuna.Trial
    fixed: Mapping[str, Any] | None = None
    choices: Mapping[str, Any] | None = None
    prefix: str = ""

    def __post_init__(self):
        self.fixed = dict(self.fixed or {})
        self.choices = dict(self.choices or {})

    def _name(self, key: str) -> str:
        return f"{self.prefix}.{key}" if self.prefix else key

    def cat(self, key: str, default: Sequence[Any]) -> Any:
        if key in self.fixed:
            return self.fixed[key]
        opts = self.choices.get(key, default)
        return self.trial.suggest_categorical(self._name(key), list(opts))

    def flt(self, key: str, low: float, high: float, *, log: bool = False) -> float:
        if key in self.fixed:
            return float(self.fixed[key])
        spec = self.choices.get(key, (low, high, log))
        if isinstance(spec, tuple):
            if len(spec) == 2:
                low_, high_ = spec
                log_ = log
            else:
                low_, high_, log_ = spec
        else:
            # allow passing dict-like specs if you want later
            raise TypeError(f"Bad choices spec for {key}: {spec}")
        return float(
            self.trial.suggest_float(
                self._name(key), float(low_), float(high_), log=bool(log_)
            )
        )

    def int(self, key: str, low: int, high: int, *, log: bool = False) -> int:
        if key in self.fixed:
            return int(self.fixed[key])
        spec = self.choices.get(key, (low, high, log))
        if len(spec) == 2:
            low_, high_ = spec
            log_ = log
        else:
            low_, high_, log_ = spec
        return int(
            self.trial.suggest_int(
                self._name(key), int(low_), int(high_), log=bool(log_)
            )
        )


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
    def sample(
        cls,
        trial: optuna.Trial,
        *,
        fixed: Mapping[str, Any] | None = None,
        choices: Mapping[str, Any] | None = None,
        prefix: str = "unet",
    ) -> "UNetConfig":
        s = SamplerCtx(trial, fixed=fixed, choices=choices, prefix=prefix)

        channels = s.cat(
            "channels",
            [(1, 64, 128), (1, 64, 128, 256), (1, 64, 128, 256, 512)],
        )
        return cls(
            channels=tuple(channels),
            d_trunk=int(s.cat("d_trunk", [16, 32, 64])),
            d_concat=int(s.cat("d_concat", [4, 8, 16])),
            group_norm_size=int(s.cat("group_norm_size", [4, 8])),
            d_time=int(s.cat("d_time", [64, 128])),
            max_time_period=float(s.cat("max_time_period", [1000.0, 10000.0])),
            activation=s.cat("activation", ["silu", "relu", "gelu"]),
            upsample_mode=s.cat(
                "upsample_mode", ["nearest", "bilinear", "convtranspose"]
            ),
        )

    def to_mlflow_params(self, *, prefix: str = "unet") -> dict[str, Any]:
        p = {
            "channels": ",".join(map(str, self.channels)),
            "d_trunk": self.d_trunk,
            "d_concat": self.d_concat,
            "group_norm_size": self.group_norm_size,
            "d_time": self.d_time,
            "max_time_period": self.max_time_period,
            "activation": self.activation,
            "upsample_mode": self.upsample_mode,
        }

        if prefix:
            return {f"{prefix}.{k}": v for k, v in p.items()}
        return p


@dataclass(frozen=True)
class OptimConfig:
    name: OptimName = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4

    @classmethod
    def sample(
        cls,
        trial: optuna.Trial,
        *,
        fixed: Mapping[str, Any] | None = None,
        choices: Mapping[str, Any] | None = None,
        prefix: str = "optim",
    ) -> "OptimConfig":
        s = SamplerCtx(trial, fixed=fixed, choices=choices, prefix=prefix)

        name = s.cat("name", ["adamw", "adam"])
        lr = s.flt("lr", 1e-5, 5e-4, log=True)
        wd = s.flt("weight_decay", 1e-6, 1e-2, log=True)

        # # only used if you decide to tune them
        # beta1 = s.flt("beta1", 0.85, 0.95) if name in ("adamw", "adam") else 0.9
        # beta2 = s.flt("beta2", 0.98, 0.999) if name in ("adamw", "adam") else 0.999
        # momentum = s.flt("momentum", 0.0, 0.95) if name == "sgd" else 0.9

        return cls(
            name=name,
            lr=lr,
            weight_decay=wd,
            # beta1=beta1,
            # beta2=beta2,
            # momentum=momentum,
        )

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

    def to_mlflow_params(self, *, prefix: str = "optim") -> dict[str, Any]:
        p = {
            "name": self.name,
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
        }
        # if self.name in ("adam", "adamw"):
        #     p.update({
        #         "beta1": self.beta1,
        #         "beta2": self.beta2,
        #     })

        # if self.name == "sgd":
        #     p["momentum"] = self.momentum

        if prefix:
            return {f"{prefix}.{k}": v for k, v in p.items()}
        return p



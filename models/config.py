from dataclasses import dataclass, is_dataclass, fields
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Tuple, Literal, Mapping, Any, Sequence
import torch.nn as nn
import torch.optim
import optuna

OptimName = Literal["adam", "adamw", "sgd"]
ActivationName = Literal["relu", "silu", "gelu"]
UpsampleMode = Literal["nearest", "bilinear", "convtranspose"]

def _mlflow_value(v: Any, *, seq_sep: str = ",") -> Any:
    """Convert a value into something MLflow log_params can handle nicely."""
    if v is None:
        return "null"

    if isinstance(v, (list, tuple)):
        return seq_sep.join(map(str, v))

    if isinstance(v, Mapping):
        # You can choose to raise instead if you want strictness
        return str(dict(v))

    # Most scalars are fine (int/float/bool/str)
    return v

def dataclass_to_mlflow_params(obj: Any, *, prefix: str = "", seq_sep: str = ",") -> dict[str, Any]:
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass instance, got {type(obj)}")

    out: dict[str, Any] = {}
    for f in fields(obj):
        k = f.name
        v = getattr(obj, k)
        out[k] = _mlflow_value(v, seq_sep=seq_sep)

    if prefix:
        return {f"{prefix}.{k}": v for k, v in out.items()}
    return out

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



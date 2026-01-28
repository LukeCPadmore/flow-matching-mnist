from typing import Any, Dict, List, Literal, Optional, Union, Mapping
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, model_validator
import optuna
import torchvision.transforms as transforms
from utils.create_dataloaders import default_transform

class FloatSpec(BaseModel):
    type: Literal["float"]
    low: float
    high: float
    log: bool = False
    step: Optional[float] = None

    @model_validator(mode="after")
    def _check_bounds(self):
        if not (self.low < self.high):
            raise ValueError(
                f"float spec requires low < high (got {self.low} >= {self.high})"
            )
        if self.log and self.low <= 0:
            raise ValueError("log=true requires low > 0 for float spec")
        return self


class IntSpec(BaseModel):
    type: Literal["int"]
    low: int
    high: int
    log: bool = False
    step: int = 1

    @model_validator(mode="after")
    def _check_bounds(self):
        if not (self.low < self.high):
            raise ValueError(
                f"int spec requires low < high (got {self.low} >= {self.high})"
            )
        if self.step <= 0:
            raise ValueError("int spec requires step > 0")
        if self.log and self.low <= 0:
            raise ValueError("log=true requires low > 0 for int spec")
        return self


class CategoricalSpec(BaseModel):
    type: Literal["categorical"]
    choices: List[Any]

    @model_validator(mode="after")
    def _check_choices(self):
        if len(self.choices) == 0:
            raise ValueError("categorical spec requires non-empty choices")
        return self


SearchSpec = FloatSpec | IntSpec | CategoricalSpec


class UNetHPT(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: dict[str, Any] = Field(default_factory=dict)
    choices: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fixed(self):
        # Optional: validate known fixed keys/types
        if "activation" in self.fixed:
            if self.fixed["activation"] not in ("relu", "silu", "gelu"):
                raise ValueError("unet.fixed.activation must be one of relu|silu|gelu")
        if "upsample_mode" in self.fixed:
            if self.fixed["upsample_mode"] not in (
                "nearest",
                "bilinear",
                "convtranspose",
            ):
                raise ValueError(
                    "unet.fixed.upsample_mode must be nearest|bilinear|convtranspose"
                )
        return self


class OptimHPT(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed: dict[str, Any] = Field(default_factory=dict)
    choices: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self):
        if "name" in self.fixed and self.fixed["name"] not in ("adam", "adamw", "sgd"):
            raise ValueError("optim.fixed.name must be adam|adamw|sgd")
        return self

class OptunaStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    study_name: str = "optuna_study"
    direction: Literal["minimize","maximize"] = Field(default = "minimize")
    n_trials: int = Field(default = 1, ge = 1)
    storage: str | Path = "sqlite:///optuna.db"

class DataloaderConfig(BaseModel): 
    model_config = ConfigDict(extra="forbid")
    data_path: str | Path = "/home/luke-padmore/Source/flow-matching-mnist/data"
    batch_size: int = Field(default = 64, ge = 1)
    num_workers: int  = Field(default = 0, ge = 0)
    transform: transforms.Compose = default_transform
    num_workers: int  = Field(default = 0, ge = 0, le = 4)
    shuffle: bool = True

class HPTYaml(BaseModel):
    model_config = ConfigDict(extra="forbid")
    optuna_study: OptunaStudy = Field(default_factory=OptunaStudy)
    unet: UNetHPT = Field(default_factory=UNetHPT)
    optim: OptimHPT = Field(default_factory=OptimHPT)

def _sample_one(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    # categorical
    if "cat" in spec:
        return trial.suggest_categorical(name, list(spec["cat"]))

    # float range
    if "float" in spec:
        low, high = spec["float"]
        return trial.suggest_float(name, float(low), float(high), log=bool(spec.get("log", False)))

    # int range
    if "int" in spec:
        low, high = spec["int"]
        return trial.suggest_int(name, int(low), int(high), log=bool(spec.get("log", False)), step=int(spec.get("step", 1)))

    raise ValueError(f"Unknown choice spec for {name}: {spec}")
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


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


SearchSpec = Union[FloatSpec, IntSpec, CategoricalSpec]

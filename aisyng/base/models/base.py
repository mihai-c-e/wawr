from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel


class PayloadBase(BaseModel):
    @classmethod
    def model_validate_or_none(cls, model_dict: Dict[str, Any]) -> PayloadBase | None:
        raise NotImplementedError()
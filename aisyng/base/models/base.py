from __future__ import annotations
from typing import Dict, Any, Optional
from pydantic import BaseModel


class PayloadBase(BaseModel):
    loaded_values: Dict[str, Any]
    meta: Dict[str, Any]
    type_id: Optional[str] = None

    @classmethod
    def create_payload_object_from_graph_element_dict(cls, data: Dict[str, Any]) -> PayloadBase | None:
        raise NotImplementedError()

    def __init__(self, **kwargs):
        kwargs['meta'] = kwargs.get('meta') or dict()
        kwargs['loaded_values'] = dict(kwargs)
        super().__init__(**kwargs)
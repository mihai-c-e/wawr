from enum import Enum
from typing import Optional, List, Any, Callable, Dict
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, field_validator, field_serializer
from aisyng.base.utils import strptime_admyhmsgmt, strptime_ymdhms, strftime_ymdhms, strftime_ymd, strptime_ymd
from aisyng.base.models import GraphElement


def _validate_date(obj: Any, date_validators: List[Callable]) -> datetime:
    if isinstance(obj, datetime):
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    for validator in date_validators:
        try:
            return validator(obj)
        except Exception:
            pass
    raise ValueError(f"Can't treat value of type: {type(obj)}, or all validators failed: {obj}")


class GraphElementTypes(str, Enum):
    Abstract = "abstract"
    Title = "title"
    Fact = "fact",
    FactType = "fact_type",
    Entity = "entity",
    EntityType = "entity_type"

    IsTitleOf = "is_title_of"
    IsExtractedFrom = "is_extracted_from"
    IsOfType = "is_of_type"
    IsA = "is_a"

def should_ignore_graph_element_duplicates(ge: GraphElement):
    return ge.type_id in [
        GraphElementTypes.FactType,
        GraphElementTypes.EntityType,
        GraphElementTypes.Entity
    ]


class PaperVersions(BaseModel):
    version: str
    created: datetime

    @field_validator("created", mode='before')
    def validate_date(cls, obj: Any) -> datetime:
        return _validate_date(obj, date_validators=[strptime_admyhmsgmt, strptime_ymdhms])

    @field_serializer("created")
    def serialize_date(self, d: datetime):
        return strftime_ymdhms(d)


class PaperAbstract(BaseModel):
    id: str
    title: str
    abstract: str
    date: datetime
    submitter: Optional[str] = None
    authors: Optional[str] = None
    comments: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    report_no: Optional[str] = None
    categories: Optional[str] = None
    license: Optional[str] = None
    versions: List[PaperVersions] = None
    update_date: Optional[datetime] = None
    authors_parsed: List[List[str]] = list()

    @field_validator("date", mode='before')
    def validate_date(cls, obj: Any) -> datetime:
        return _validate_date(obj, date_validators=[strptime_ymdhms])

    @field_validator("update_date", mode='before')
    def validate_update_date(cls, obj: Any) -> datetime:
        return _validate_date(obj, date_validators=[strptime_ymd])

    @field_serializer("date")
    def serialize_date(self, d: datetime) -> str:
        return strftime_ymdhms(d)

    @field_serializer("update_date")
    def serialize_update_date(self, d: datetime) -> str:
        return strftime_ymd(d)

class Fact(BaseModel):
    type: str
    text: str
    citation: str
    date: datetime

    @classmethod
    def model_validate_with_date(cls, fact_dict: Dict[str, Any], date: datetime) -> "Fact":
        fact_dict['date'] = date
        return cls.model_validate(fact_dict)
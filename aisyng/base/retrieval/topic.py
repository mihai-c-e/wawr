from datetime import datetime
from typing import Callable, List, Any, Optional, Dict
from pydantic import BaseModel
from aisyng.base.models import GraphNode, GraphRelationship

_date_format = '%d-%m-%Y %H:%M:%SZ'

class TopicBreakdown(BaseModel):
    filter: List[str]
    questions: List[str]

class TopicMeta(BaseModel):
    source_id: str
    progress: float = 0.0
    status: str = "initialised"
    distance_threshold: float = 0.7
    subgraph_ids: List[str] = list()
    reference_ids: List[str] = list()
    reference_scores: List[float] = list()
    log_history: List[str] = list()
    embedding_key: str
    model: str
    limit: int = 1000
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    response: str = ""
    usage: Optional[Dict[str, Any]] = None
    user_message: str = ""
    breakdown: Dict[str, Any] = dict()
    # hypothetical: List[str] = list()

    @classmethod
    def date_to_str(cls, d: datetime):
        return d.strftime(_date_format)

    def get_from_date(self) -> datetime:
        if self.from_date is None or self.from_date == "":
            return None
        return datetime.strptime(self.from_date, _date_format)

    def set_from_date(self, d: datetime):
        self.from_date = d.strftime(_date_format)

    def get_to_date(self) -> datetime:
        if self.to_date is None or self.to_date == "":
            return None
        return datetime.strptime(self.to_date, _date_format)

    def set_to_date(self, d: datetime):
        self.to_date = d.strftime(_date_format)

class TopicNode(GraphNode):
    def __init__(self, text: str = None, topic_meta: TopicMeta = None, **kwargs):
        if 'meta' in kwargs and topic_meta is not None:
            raise ValueError("Provide either topic_meta as TopicMeta or meta as dict")
        meta = dict(topic_meta) if topic_meta is not None else kwargs.pop('meta')
        kwargs.pop('type_id', None)
        super().__init__(text, meta=meta, type_id="Topic", **kwargs)

    def get_topic_meta(self) -> TopicMeta:
        return TopicMeta.model_validate(self.meta)

    def update_topic_meta(self, topic_meta: TopicMeta) -> None:
        #self.meta.update(dict(topic_meta))
        self.meta = topic_meta

class TopicMatchRelationship(GraphRelationship):
    def __init__(self, score: float = None, match_type: str = None, **kwargs):
        meta = kwargs.pop("meta", dict())
        if score is None and meta.get("score") is None:
            raise ValueError("Missing similarity score")
        if match_type is None and meta.get("match_type") is None:
            raise ValueError("Missing match type")

        if score is not None:
            meta["score"] = score
        if match_type is not None:
            meta["match_type"] = match_type
        kwargs["type_id"] = "similar_to"
        kwargs["meta"] = meta

        super().__init__(text="matches", **kwargs)

class TopicReference(BaseModel):
    index: int
    text: str
    fact_type: str
    citation: str
    date: datetime
    title: str
    url: str
    similarity: float
 

class TopicSolverBase:

    def solve(self, node: TopicNode):
        raise NotImplementedError()


from datetime import datetime
from typing import Callable, List, Any, Optional, Dict
from pydantic import BaseModel
from ekb.base.models import GraphNode, GraphRelationship

class TopicMeta(BaseModel):
    source_id: str
    progress: float = 0.0
    status: str = "initialised"
    distance_threshold: float = 0.7
    subgraph_ids: List[str] = list()
    reference_ids: List[str] = list()
    log_history: List[str] = list()
    embedding_key: str
    model: str
    limit: int = 1000
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    response: str = ""
    usage: Optional[Dict[str, Any]] = None

class TopicNode(GraphNode):
    def __init__(self, text: str, topic_meta: TopicMeta):
        super().__init__(text, meta=dict(topic_meta), type_id="Topic")

    def get_topic_meta(self) -> TopicMeta:
        return TopicMeta.model_validate(self.meta)

    def update_topic_meta(self, topic_meta: TopicMeta) -> None:
        self.meta.update(dict(topic_meta))

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
        kwargs["meta"] = meta

        super().__init__(text="matches", **kwargs)


class TopicSolverBase:

    def solve(self, node: TopicNode):
        raise NotImplementedError()


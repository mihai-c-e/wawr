from datetime import datetime
from typing import Callable, List, Any, Optional, Dict
from pydantic import BaseModel, field_validator, field_serializer
from aisyng.base.models import GraphNode, GraphRelationship
from aisyng.base.utils import strptime_ymdhms, strftime_ymdhms
from aisyng.wawr.models._models_utils import _validate_date
from aisyng.base.context import AppContext
from aisyng.base.llms.base import LLMName, InappropriateContentException


class TopicBreakdown(BaseModel):
    filter: List[str]
    questions: List[str]

class TopicMeta(BaseModel):
    source_id: str
    ask: str


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

class TopicSolverBase(BaseModel):
    progress: float = 0.0
    status: str = "initialised"
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    log_history: List[str] = list()
    usage: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = ""
    answer: Optional[str] = ""

    @field_validator("from_date", "to_date", mode='before')
    def validate_date(cls, obj: Any) -> datetime:
        return _validate_date(obj=obj, date_validators=[strptime_ymdhms])

    @field_serializer("from_date", "to_date")
    def serialize_date(self, d: datetime):
        return strftime_ymdhms(d)

    def solve_internal(self, topic_node: TopicNode, context: AppContext):
        raise NotImplementedError()

    def solve(self, topic_node: TopicNode, context: AppContext):
        try:
            llm = context.llm_providers.get_by_model_name(self.llm_name)
            llm.moderate_text(text=topic_node.text)
            self.solve_internal(topic_node=topic_node, context=context)
        except InappropriateContentException as ex:
            self.status = "Error"
            self.progress = 0.0
            self.log_history.append(f"Error: {str(ex)}")
            self.user_message = "Your question was flagged as inappropriate and will not be processed."
            #TODO: persist

class DirectSimilarityTopicSolver(TopicSolverBase):
    distance_threshold: float = 0.7
    embedding_key: str
    llm_name: LLMName
    limit: int = 1000

    def solve_internal(self, topic_node: TopicNode, context: AppContext):
        llm = context.llm_providers.get_by_model_name(self.llm_name)
        context.add_graph_elements_embedding(embedding_key=self.embedding_key, element=topic_node)


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


from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Iterable

from jinja2 import Template
from pydantic import BaseModel, field_validator, field_serializer

from aisyng.base.context import AppContext
from aisyng.base.llms.base import InappropriateContentException, LLMName
from aisyng.base.utils import strptime_ymdhms, strftime_ymdhms, _validate_date
from aisyng.base.models.graph import ScoredGraphElement, GraphElementTypes, GraphNode, GraphRelationship, GraphElement
from aisyng.base.models.base import PayloadBase


class TopicMeta(PayloadBase):
    source_id: str
    ask: str
    type_id: str = GraphElementTypes.Topic

    @classmethod
    def model_validate_or_none(cls, model_dict: Dict[str, Any]) -> PayloadBase | None:
        if model_dict.get("type_id") == GraphElementTypes.Topic:
            return cls.model_validate(model_dict)
        return None


class TopicSolverCallback:
    def state_changed(self, topic_solver: TopicSolverBase) -> None:
        raise NotImplementedError()


class TopicSolverBase(PayloadBase):
    _callbacks: List[TopicSolverCallback] = list()
    progress: float = 0.0
    status: str = "initialised"
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    log_history: List[str] = list()
    usage: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = ""
    answer: Optional[str] = ""


    @field_validator("from_date", "to_date", mode='before')
    def validate_date(cls, obj: Any) -> datetime:
        return _validate_date(obj=obj, date_validators=[strptime_ymdhms])

    @field_serializer("from_date", "to_date")
    def serialize_date(self, d: datetime):
        return None if d is None else strftime_ymdhms(d)

    def _update_state(
            self,
            status: str = None,
            progress: float = None,
            log_entry: str = None,
            user_message: str = None
    ):
        logging.info(
            f"Updating job: status={status}, progress={progress}, user_message={user_message}, log_entry={log_entry}"
        )
        self.status = status or self.status
        self.progress = progress or self.progress
        self.user_message = user_message or self.user_message
        if log_entry:
            self.log_history.append(f"{strftime_ymdhms(datetime.now())}: {log_entry}")

        for callback in self._callbacks:
            callback.state_changed(self)

    def add_callback(self, callback: TopicSolverCallback):
        self._callbacks.append(callback)

    def solve_internal(self, ask: str, ask_embeddings: List[float], context: AppContext, **kwargs):
        raise NotImplementedError()

    def solve(self, ask: str, ask_embedding: List[float] | None, context: AppContext, **kwargs):
        try:
            llm = context.llm_providers.get_by_model_name(self.llm_name)

            self._update_state(status="moderating", progress=0.0)
            llm.moderate_text(text=ask)

            self._update_state(status="solving", progress=0.1, log_entry="Moderation passed")
            self.solve_internal(ask=ask, ask_embedding=ask_embedding, context=context, **kwargs)

        except InappropriateContentException as ex:
            self._update_state(
                status="Inappropriate content",
                progress=0.0,
                user_message="Your question was flagged as inappropriate and will not be processed.",
                log_entry=f"Error: {str(ex)}"
            )
        except Exception as ex:
            self._update_state(
                status="Internal error",
                progress=0.0,
                user_message="There was an error processing your request. "
                             "We apologise for the inconvenience, please try again.",
                log_entry=f"Error: {str(ex)}"
            )
            raise


class DirectSimilarityTopicSolverBase(TopicSolverBase):
    _context: AppContext
    distance_threshold: float = 0.7
    embedding_key: str
    llm_name: LLMName
    limit: int = 1000
    answer: Optional[str] = None
    prompt_template: str
    solver_type: str="direct_similarity"

    @classmethod
    def model_validate_or_none(cls, model_dict: Dict[str, Any]) -> PayloadBase | None:
        if (model_dict.get("type_id") == GraphElementTypes.TopicSolver
                and model_dict.get('solver_type') == "direct_similarity"):
            return cls.model_validate(model_dict)
        return None

    def query_model(self, question: str, references: str, **kwargs) -> str:
        raise NotImplementedError()

    def nodes_to_references_prompt_part(self, nodes: List[GraphNode]):
        raise NotImplementedError()

    def refine_search(self, matched_nodes: List[ScoredGraphElement]) -> List[List[GraphElement]]:
        raise NotImplementedError()

    def select_references(self, graph_elements: Iterable[GraphElement]) -> List[GraphNode]:
        raise NotImplementedError()

    def find_by_similarity(self, ask: str, ask_embedding: List[float]) -> List[ScoredGraphElement]:
        return self._context.persistence.find_by_similarity(
            with_strings=[ask],
            with_vectors=[ask_embedding],
            distance_threshold=self.distance_threshold,
            embedder=self._context.embedding_pool.get_embedder(self.embedding_key),
            limit=self.limit,
            from_date=self.from_date,
            to_date=self.to_date,
        )

    def solve_internal(self, ask: str, ask_embedding: List[float], context: AppContext, **kwargs):
        self._context = context
        if ask_embedding is None:
            self._update_state(status="embedding", progress=0.1)
            ask_embedding = context.embedding_pool.get_embedder(self.embedding_key).create_embeddings([ask])[0]

        self._update_state(status="retrieving", progress=0.2, log_entry="Embeddings calculated")
        matched_nodes: List[ScoredGraphElement] = self.find_by_similarity(ask=ask, ask_embedding=ask_embedding)

        self._update_state(status="refining", progress=0.5, log_entry="Initial matching complete")
        matched_paths: List[List[GraphElement]] = self.refine_search(matched_nodes=matched_nodes)

        self._update_state(status="building references", progress=0.5, log_entry="Refined matching complete")
        graph_elements = {element for element_list in matched_paths for element in element_list}
        reference_nodes = self.select_references(graph_elements=graph_elements)
        reference_nodes = sorted(reference_nodes, key=lambda x: x.date)[::-1]
        references_as_text = self.nodes_to_references_prompt_part(reference_nodes)

        self._update_state(status="generating answer", progress=0.8, log_entry="References build complete")
        answer = self.query_model(question=ask, references=references_as_text)
        self.answer = answer
        self._update_state(status="completed", progress=1.0, log_entry="Answer complete")

        return self



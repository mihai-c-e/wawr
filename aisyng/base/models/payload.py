from __future__ import annotations

import concurrent.futures
import logging
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Iterable, Callable, cast, Tuple

from jinja2 import Template
from pydantic import BaseModel, field_validator, field_serializer

from aisyng.base.context import AppContext
from aisyng.base.llms.base import InappropriateContentException, LLMName
from aisyng.base.utils import strptime_ymdhms, strftime_ymdhms, _validate_date
from aisyng.base.models.graph import ScoredGraphElement, GraphElementTypes, GraphNode, GraphRelationship, GraphElement
from aisyng.base.models.base import PayloadBase


class TopicSolverStatus(str, Enum):
    Initialised = "initialised"
    Completed = "completed"
    InternalError = "internal error"
    InappropriateContent = "inappropriate content"
    Retrieving = "retrieving"
    Refining = "refining"
    RetrievingGraph="retrieving graph"
    GeneratingAnswer = "generating answer"
    BuildingReferences = "building references"
    Embedding = "embedding"
    Moderating = "moderating"
    Solving = "solving"


_metric_extration_template = Template(
    "Title:\"{{ title }}\"\n"
    "Abstract: \"{{ abstract }}\"\n\n"
    "Based on the abstract above, infer and justify an answer to this ask: "
    "\"{{ ask }}\". \n"
    "Answer in json format with the following keys, using as few words as possible. Also use metric values wherever possible: \n"
    "{{ output_format }}\n"
)

_reference_answer_template = Template(
    "Title:\"{{ title }}\"\n"
    "Abstract: \"{{ abstract }}\"\n\n"
    "Based on the abstract above, infer and justify an answer to this ask: "
    "\"{{ ask }}\". "
    "If the abstract is not informative for answering the question, output just \"irrelevant\", without any justification."
)

_summary_table_schema_template = Template(
    'Input: "Does GPT-3.5 outperform GPT-4 on any task?"\n'
    'Output: ["Task", "Task domain", "GPT-3.5 Score", "GPT-4 Score", "Winner"]\n\n'

    'Input: "How are language models applied in medicine?"\n'
    'Output: ["Model", "Medical domain", "Application", "Performance"]\n'

    'In the examples above you are given a question which needs to be answered based on information '
    'from multiple documents. We aim to not only answer with text, but display a table or chart with '
    'structured data as results. Think what relevant data could be collected from the texts, in '
    'relation to the question, in table format. Provide the table columns as a json array,  as per '
    'the examples above.\n\n'

    'Input: "{{ ask }}"\n'
    'Output: '
)


class TopicMeta(PayloadBase):
    source_id: str
    ask: str
    type_id: str = GraphElementTypes.Topic

    @classmethod
    def create_payload_object_from_graph_element_dict(cls, data: Dict[str, Any]) -> PayloadBase | None:
        meta = data.get("meta")
        if meta is None:
            return None
        if meta["type_id"] == GraphElementTypes.Topic:
            return cls(**meta)
        return None


class AnswerWithReferencesFromNodes:
    context: AppContext
    llm_name: LLMName
    nodes: List[GraphNode]
    ask: str

    def __init__(self, context: AppContext, llm_name: LLMName, nodes: List[GraphNode], ask: str):
        self.context = context
        self.llm_name = llm_name
        self.nodes = nodes
        self.ask = ask

    def nodes_to_references_prompt_part(self, **kwargs):
        raise NotImplementedError()

    def get_prompt(self):
        raise NotImplementedError()

    def get_answer(self, **kwargs) ->Tuple[str, Any]:
        llm_provider = self.context.llm_providers.get_by_model_name(self.llm_name)
        response, usage = llm_provider.query_model(
            query=self.get_prompt(),
            model=self.llm_name,
            **kwargs
        )
        return response, usage


class TopicSolverCallback:
    def state_changed(self, topic_solver: TopicSolverBase) -> None:
        raise NotImplementedError()


class TopicSolverBase(PayloadBase):
    _context: AppContext
    _callbacks: List[TopicSolverCallback] = list()
    progress: float = 0.0
    status: str = "initialised"
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    log_history: List[str] = list()
    usage: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = ""
    answer: Optional[str] = ""
    final_answer_prompt_template: Optional[str] = None
    graph_nodes: Optional[List[GraphNode]] = None
    graph_relationships: Optional[List[GraphRelationship]] = None
    reference_nodes: Optional[List[GraphNode]] = None
    partial_answers: Optional[List[str]] = None
    metric_table: Optional[List[Any]] = None

    solver_type: str

    @field_validator("from_date", "to_date", mode='before')
    def validate_date(cls, obj: Any) -> datetime | None:
        if obj is None:
            return None
        return _validate_date(obj=obj, date_validators=[strptime_ymdhms])

    @field_serializer("from_date", "to_date")
    def serialize_date(self, d: datetime):
        return None if d is None else strftime_ymdhms(d)

    def is_finished(self) -> bool:
        return self.status in [
            TopicSolverStatus.InternalError,
            TopicSolverStatus.InappropriateContent,
            TopicSolverStatus.Completed
        ]

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

    def has_graph(self) -> bool:
        return self.graph_elements is not None and len(self.graph_elements) > 0

    def has_references(self) -> bool:
        return self.reference_nodes is not None and len(self.reference_nodes) > 0

    def has_answer(self) -> bool:
        return self.answer is not None and self.answer != ""

    def add_callback(self, callback: TopicSolverCallback):
        self._callbacks.append(callback)

    def create_final_answer(self, context: AppContext, ask: str, **kwargs) -> Tuple[str, Any]:
        raise NotImplementedError()

    def solve_internal(self, ask: str, ask_embeddings: List[float], context: AppContext, **kwargs):
        raise NotImplementedError()

    def query_model_on_nodes(
            self,
            nodes: List[GraphNode],
            node_transformer: Callable[[GraphNode], str],
            model: LLMName,
            temperature: float,
            parallelism: int = 50,
            json: bool = False,
            retries: int = 5,
            **kwargs
    ):
        return self._context.llm_providers.get_by_model_name(
            model
        ).query_model_threaded(
            data=nodes,
            preprocess_fn=node_transformer,
            model=model,
            temperature=temperature,
            parallelism=parallelism,
            json = json,
            retries=retries,
            **kwargs
        )

    def query_summary_table_schema(self, ask: str, model: LLMName, temperature: float, **kwargs) -> List[str]:
        llm = self._context.llm_providers.get_by_model_name(model)
        query = _summary_table_schema_template.render(ask=ask)
        result = llm.query_model_validate_json(query=query, model=model, temperature=temperature, **kwargs)
        return result[0]

    def solve(self, ask: str, ask_embedding: List[float] | None, context: AppContext, **kwargs):
        try:
            llm = context.llm_providers.get_by_model_name(self.llm_name)

            self._update_state(status=TopicSolverStatus.Moderating, progress=0.0)
            llm.moderate_text(text=ask)

            self._update_state(status=TopicSolverStatus.Solving, progress=0.1, log_entry="Moderation passed")
            self.solve_internal(ask=ask, ask_embedding=ask_embedding, context=context, **kwargs)

        except InappropriateContentException as ex:
            self._update_state(
                status=TopicSolverStatus.InappropriateContent,
                progress=0.0,
                user_message="Your question was flagged as inappropriate and will not be processed.",
                log_entry=f"Error: {str(ex)}"
            )
        except Exception as ex:
            self._update_state(
                status=TopicSolverStatus.InternalError,
                progress=0.0,
                user_message="There was an error processing your request. "
                             "We apologise for the inconvenience, please try again.",
                log_entry=f"Error: {str(ex)}"
            )
            raise

class SolvedExternallyTopicSolver(TopicSolverBase):

    def __init__(self, **kwargs):
        kwargs["solver_type"] = "solved_externally"
        super().__init__(**kwargs)

    @classmethod
    def create_payload_object_from_graph_element_dict(cls, data: Dict[str, Any]) -> PayloadBase | None:
        meta = data.get("meta")
        if meta is None:
            return None
        if meta["type_id"] == GraphElementTypes.TopicSolver and meta.get("solver_type") == "solved_externally":
            return cls(**meta)
        return None

    def solve(self, ask: str, ask_embedding: List[float] | None, context: AppContext, **kwargs):
        if not "answer" in kwargs:
            raise ValueError("SolvedExternallyTopicSolver requires an answer to be passed directly in the solve method")
        answer = kwargs["answer"]
        self.status = TopicSolverStatus.Completed
        self.progress = 1.0
        self.answer = answer

class DirectSimilarityTopicSolverBase(TopicSolverBase):
    distance_threshold: float = 0.7
    embedding_key: str
    llm_name: LLMName
    limit: int = 1000
    include_table: bool = False

    @classmethod
    def create_payload_object_from_graph_element_dict(cls, data: Dict[str, Any]) -> PayloadBase | None:
        meta = data.get("meta")
        if meta is None:
            return None
        if meta["type_id"] == GraphElementTypes.TopicSolver and meta.get("solver_type") == "direct_similarity":
            return cls(**meta)
        return None

    def nodes_with_substitute_content_to_references_prompt_part(self, nodes: List[GraphNode], content: List[str]):
        raise NotImplementedError()

    def refine_search(self, matched_nodes: List[GraphNode]) -> List[List[GraphElement]]:
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
            exclude_text_types=["motivation"],
            exclude_type_ids=["entity"]
        )

    def solve_internal(self, ask: str, ask_embedding: List[float], context: AppContext, **kwargs):
        def per_node_answers_template_fn(data):
            return _reference_answer_template.render(
                title=data.title,
                abstract=data.text,
                ask=ask,
            )
        def per_node_metrics_template_fn(data):
            return _metric_extration_template.render(
                title=data.title,
                abstract=data.text,
                ask=ask,
                output_format=summary_table_schema
            )
        self._context = context
        if ask_embedding is None:
            self._update_state(status=TopicSolverStatus.Embedding, progress=0.1)
            ask_embedding = context.embedding_pool.get_embedder(self.embedding_key).create_embeddings([ask])[0]

        with (concurrent.futures.ThreadPoolExecutor() as executor):
            self._update_state(status=TopicSolverStatus.Retrieving, progress=0.2, log_entry="Embeddings calculated")
            matched_nodes_scored: List[ScoredGraphElement] = (
                executor.submit(self.find_by_similarity, ask=ask, ask_embedding=ask_embedding)
            )
            if self.include_table:
                summary_table_schema = executor.submit(
                    self.query_summary_table_schema,
                    ask=ask,
                    model=self.llm_name,
                    temperature=0.1
                )
        matched_nodes_scored = matched_nodes_scored.result()
        if self.include_table:
            summary_table_schema = summary_table_schema.result()
            logging.info(f"Summary table schema: {summary_table_schema}")

        logging.info(f"Found {len(matched_nodes_scored)} matching nodes")

        matched_nodes = cast(
            List[GraphNode],
            [node.element for node in matched_nodes_scored]
        )

        self._update_state(status=TopicSolverStatus.Refining, progress=0.4, log_entry="Initial matching complete")
        logging.info(f"Getting partial answers on {len(matched_nodes)} nodes")
        reference_model_responses = self.query_model_on_nodes(
            nodes=matched_nodes,
            node_transformer=per_node_answers_template_fn,
            model=LLMName.OPENAI_GPT_35_TURBO,
            temperature=0.1,
            parallelism=400
        )

        matched_nodes_filtered = [node for i, node in enumerate(matched_nodes) if
                                'irrelevant' not in reference_model_responses[i][0].lower()]

        user_msg = ""
        if len(matched_nodes_filtered) > 0:
            if len(matched_nodes_filtered) == self.limit:
                user_msg = ("There could be more data points relevant to your query in the knowledge base. "
                            "You might want to narrow down the timeframe, increase precision, or ask "
                            "a more focused question.")

            self._update_state(status=TopicSolverStatus.RetrievingGraph, progress=0.6, log_entry="Initial matching complete")
            #with concurrent.futures.ThreadPoolExecutor() as executor:

            #matched_paths = matched_paths.result()
            #self.metric_table=[m[0] for m in metrics_list.result()]
            matched_paths: List[List[GraphElement]] = self.refine_search(
                matched_nodes=matched_nodes_filtered
            )

            self.graph_nodes = list(
                {node for node_list in matched_paths for node in node_list if isinstance(node, GraphNode)}
            )
            self.graph_relationships = list(
                {rel for rel_list in matched_paths for rel in rel_list if isinstance(rel, GraphRelationship)}
            )
            logging.info(
                f"The graph has {len(self.graph_nodes)} nodes and {len(self.graph_relationships)} relationships")

            self._update_state(status=TopicSolverStatus.BuildingReferences, progress=0.8,
                               log_entry="Refined matching complete")
            reference_nodes = self.select_references(graph_elements=self.graph_nodes)
            self.reference_nodes = sorted(reference_nodes, key=lambda x: x.date)[::-1]
            logging.info(f"selected {len(reference_nodes)} reference nodes")
            self._update_state(status=TopicSolverStatus.GeneratingAnswer, progress=0.8,
                               log_entry="References build complete")

            answer, usage = self.create_final_answer(context=context, ask=ask)
            self.answer = answer

            if self.include_table:
                metrics_list = self.query_model_on_nodes(
                    nodes=self.reference_nodes,
                    node_transformer=per_node_metrics_template_fn,
                    model=LLMName.OPENAI_GPT_4o,
                    temperature=0.1,
                    parallelism=400,
                    json=True,
                    retries=5
                )
                self.metric_table = [m[0] for m in metrics_list]

        else:
            self.answer = ("No data points relevant to your query were found in the knowledge base. "
                           "Try to lower precision,"
                           " extend the timeframe, or re-formulate your question.")

        self._update_state(
            status=TopicSolverStatus.Completed,
            progress=1.0,
            log_entry="Answer complete",
            user_message=user_msg
        )

        return self

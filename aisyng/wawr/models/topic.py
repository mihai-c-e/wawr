import logging
from datetime import datetime
from typing import Callable, List, Any, Optional, Dict

from jinja2 import Template
from pydantic import BaseModel, field_validator, field_serializer
from aisyng.base.models import GraphNode, GraphRelationship, ScoredGraphElement
from aisyng.base.utils import strptime_ymdhms, strftime_ymdhms, strftime_ymd
from aisyng.wawr.models._models_utils import _validate_date
from aisyng.wawr.models.kb import GraphElementTypes
from aisyng.base.context import AppContext
from aisyng.base.llms.base import LLMName, InappropriateContentException


def node_to_reference_prompt_part(node: GraphNode, index: int) -> str:
    if node.type_id == GraphElementTypes.Abstract:
        return abstract_node_to_reference_prompt(node, index)
    raise NotImplementedError()

def abstract_node_to_reference_prompt(node: GraphNode, index: int) -> str:
    return f'{index}. Paper date: {strftime_ymd(node.date)}\nPaper title: "{node.title}"\nPaper abstract: "{node.text}"'

def nodes_to_reference_prompt_part(nodes: List[GraphNode]) -> str:
    return "\n\n".join(
        [node_to_reference_prompt_part(node, index+1) for index, node in enumerate(nodes)]
    )

class TopicBreakdown(BaseModel):
    filter: List[str]
    questions: List[str]

class TopicMeta(BaseModel):
    source_id: str
    ask: str

class TopicReference(BaseModel):
    index: int
    text: str
    fact_type: str
    citation: str
    date: datetime
    title: str
    url: str
    similarity: float

    @classmethod
    def from_graph_node(cls, node: GraphNode, index: int = -1) -> "TopicReference":
        if node.type_id == GraphElementTypes.Abstract:
            return cls.from_abstract(node)
        return TopicReference(
            index=index,
            text=node.text,
        )

    @classmethod
    def from_abstract(cls, node: GraphNode, index: int = -1) -> "TopicReference":
        return TopicReference(
            index=index,
            test=node.text,
            fact_type="abstract",
            citation=None,
            date=node.date,

        )

class TopicSolverBase(BaseModel):
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
        return strftime_ymdhms(d)

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

    def solve_internal(self, ask: str, ask_embeddings: List[float], context: AppContext, **kwargs):
        raise NotImplementedError()

    def solve(self, ask: str, ask_embeddings: List[float], context: AppContext, **kwargs):
        try:
            llm = context.llm_providers.get_by_model_name(self.llm_name)

            self._update_state(status="moderating", progress=0.0)
            llm.moderate_text(text=ask)

            self._update_state(status="solving", progress=0.1, log_entry="Moderation passed")
            self.solve_internal(ask=ask, ask_embeddings=ask_embeddings, context=context, **kwargs)
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

class DirectSimilarityTopicSolver(TopicSolverBase):
    _context: AppContext
    distance_threshold: float = 0.7
    embedding_key: str
    llm_name: LLMName
    limit: int = 1000
    answer: Optional[str] = None

    def get_answer(self, question: str, references: str) -> str:
        prompt_template = """
    This is a set of reference extracts from research papers, ordered by relevance:
    {{ references_as_text }}

    Based on the knowledge from the references above, infer an answer as complete as possible to the following ask: "{{ question }}".

    Address what is explicitly stated in the given references, then try to infer a conclusion. Do not output anything after the conclusion. Format the text using html tags, ready to insert as-is into a html page, using text formatting. Quote the facts above in your answer in [1][2] format. Do not list references, only use numbers in your answer to refer to the facts. Write in concise style, in British English, but be very thorough take into account all relevant references. If the references are not relevant for the ask, say so. Do not write references or bibliography at the end. Do not write references, only insert indexes towards given references.

    """
        prompt = Template(prompt_template).render(
            question=question,
            references_as_text=references
        )
        llm_provider = self._context.llm_providers.get_by_model_name(self.llm_name)
        response, usage = llm_provider.query_model(query=prompt, model=self.llm_name)
        return response

    def solve_internal(self, ask: str, ask_embeddings: List[float], context: AppContext, **kwargs):
        self._context = context
        if ask_embeddings is None:
            self._update_state(status="embedding", progress=0.1)
            ask_embeddings = context.embedding_pool.get_embedder(self.embedding_key).create_embeddings([ask])[0]

        self._update_state(status="retrieving", progress=0.2, log_entry="Embeddings calculated")
        matched_nodes: List[ScoredGraphElement] = context.persistence.find_by_similarity(
            with_strings=[ask],
            with_vectors=[ask_embeddings],
            distance_threshold=self.distance_threshold,
            embedder=context.embedding_pool.get_embedder(self.embedding_key),
            limit=self.limit,
            from_date=self.from_date,
            to_date=self.to_date,
        )

        self._update_state(status="refining", progress=0.5, log_entry="Initial matching complete")
        matched_paths = context.persistence.get_paths_between(
            from_node_ids=[mn.element.id for mn in matched_nodes],
            to_node_label=GraphElementTypes.Abstract,
            via_relationships=[GraphElementTypes.IsExtractedFrom, GraphElementTypes.IsTitleOf]
        )

        self._update_state(status="building references", progress=0.5, log_entry="Refined matching complete")
        graph_elements = {element for element_list in matched_paths for element in element_list}
        reference_nodes = [element for element in graph_elements if element.type_id == GraphElementTypes.Abstract]
        reference_nodes = sorted(reference_nodes, key=lambda x: x.date)[::-1]
        references_as_text = nodes_to_reference_prompt_part(reference_nodes)

        self._update_state(status="generating answer", progress=0.8, log_entry="References build complete")
        answer = self.get_answer(question=ask, references=references_as_text)
        self.answer = answer
        self._update_state(status="completed", progress=1.0, log_entry="Answer complete")

        return self


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


from datetime import datetime
from typing import Callable, List, Any, Optional, Dict
from pydantic import BaseModel, field_validator, field_serializer
from aisyng.base.models import GraphNode, GraphRelationship, ScoredGraphElement
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
            self.status = "Inappropriate content"
            self.progress = 0.0
            self.log_history.append(f"Error: {str(ex)}")
            self.user_message = "Your question was flagged as inappropriate and will not be processed."
            #TODO: persist
        except Exception as ex:
            self.status = "Error"
            self.progress = 0.0
            self.log_history.append(f"Error: {str(ex)}")
            self.user_message = "There was an error processing your request")
            #TODO: persist
            raise

class DirectSimilarityTopicSolver(TopicSolverBase):
    distance_threshold: float = 0.7
    embedding_key: str
    llm_name: LLMName
    limit: int = 1000

    def solve_internal(self, topic_node: TopicNode, context: AppContext):
        llm = context.llm_providers.get_by_model_name(self.llm_name)
        context.add_graph_elements_embedding(embedding_key=self.embedding_key, element=topic_node)
        self.log_history.append("Calculated embeddings")
        self.status = "Retrieving from knowledge base"
        self.progress = 0.1
        # TODO persist
        matched: List[ScoredGraphElement] = context.persistence.find_by_similarity(
            with_strings=[topic_node.text],
            with_vectors=[topic_node.embeddings[self.embedding_key]],
            distance_threshold=self.distance_threshold,
            embedder=context.embedding_pool.get_embedder(self.embedding_key),
            limit=self.limit,
            from_date=self.from_date,
            to_date=self.to_date,

        )
        relationships = self.create_similarity_relationships(node=node, embedding_key=embedding_key)
        meta.subgraph_ids = [rel.id for rel in relationships]
        meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
        meta.status = "References"
        meta.progress = 0.5
        meta.log_history.append("Retrieved subgraph")
        node.update_topic_meta(meta)
        self.sql_toolkit.persist_graph_elements(elements_merge=[node], elements_add=relationships)

        reference_nodes, reference_scores = self.identify_reference_nodes(node=node, subgraph=relationships)
        reference_nodes, reference_scores = self.limit_references(meta=meta, reference_nodes=reference_nodes,
                                                                  reference_scores=reference_scores)
        meta.reference_ids = [n.id for n in reference_nodes]
        meta.reference_scores = [s for s in reference_scores]
        references = self.create_references(node=node, reference_nodes=reference_nodes,
                                            reference_scores=reference_scores)

        meta.status = "Answering"
        meta.progress = 0.7
        meta.log_history.append("Retrieved references")
        node.update_topic_meta(meta)
        self.sql_toolkit.persist_graph_elements(elements_merge=[node])

        if len(references) == 0:
            meta.response = "No data found. Try relaxing the parameters (e.g. larger time interval, or lower precision)."
            meta.usage = {}
        else:
            response = self.get_answer(node=node, references=references)
            meta.response = response[0]
            meta.usage = dict(response[1].usage)
        meta.status = "Completed"
        meta.progress = 1.0
        meta.log_history.append("Answer received")
        node.update_topic_meta(meta)
        self.sql_toolkit.persist_graph_elements(elements_merge=[node])
        return node

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


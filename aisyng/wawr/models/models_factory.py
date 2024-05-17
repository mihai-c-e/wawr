from datetime import datetime

from aisyng.wawr.models.graph import PaperAbstract, WAWRGraphElementTypes
from aisyng.base.models.graph import GraphNode, GraphRelationship, GraphElementTypes
from base.models.payload import TopicMeta, TopicSolverBase


def get_abstract_node_id(node_info: PaperAbstract):
    return node_info.id

def get_title_node_id(node_info: PaperAbstract):
    return f"title:{node_info.id}"

def get_relationship_id(node1: GraphNode, relationship_name: str, node2: GraphNode) -> str:
    return f"{node1.id}-{relationship_name}-{node2.id}"

def create_abstract_node(abstract: PaperAbstract) -> GraphNode:
    return GraphNode(
        id=get_abstract_node_id(abstract),
        text=abstract.abstract,
        date=abstract.date,
        title=abstract.title,
        meta=abstract,
        type_id=WAWRGraphElementTypes.Abstract,
    )

def create_title_node_from_abstract_info(abstract_node: GraphNode) -> GraphNode:
    node_info = abstract_node.meta
    return GraphNode(
        id=get_title_node_id(node_info),
        text=node_info.title,
        date=node_info.date,
        title=node_info.title,
        meta=node_info,
        type_id=WAWRGraphElementTypes.Title,
        source_id=node_info.id,
        text_type='title',
    )

def create_title_to_abstract_relationship(abstract_node: GraphNode, title_node: GraphNode) -> GraphRelationship:
    return GraphRelationship(
        id=get_relationship_id(title_node, WAWRGraphElementTypes.IsTitleOf, abstract_node),
        from_node=title_node,
        to_node=abstract_node,
        text="title of",
        type_id=WAWRGraphElementTypes.IsTitleOf,
        date=abstract_node.date,
    )

def create_topic_node(ask: str, source_id: str) -> GraphNode:
    topic_meta = TopicMeta(ask=ask, source_id=source_id)
    return GraphNode(
        text=ask,
        date=datetime.now(),
        title=ask,
        meta=topic_meta,
        type_id=GraphElementTypes.Topic,
        source_id=source_id,
    )

def create_topic_solver_node(topic_node: GraphNode, topic_solver: TopicSolverBase) -> GraphNode:
    return GraphNode(
        text="",
        date=datetime.now(),
        title=f"Answering on: {topic_node.text}",
        meta=topic_solver,
        type_id=GraphElementTypes.TopicSolver,
        source_id=topic_node.id,
    )

def create_topic_solver_relationship(topic_solver_node: GraphNode, topic_node: GraphNode) -> GraphRelationship:
    return GraphRelationship(
        from_node=topic_node,
        to_node=topic_solver_node,
        text=GraphElementTypes.IsSolvedBy,
        type_id=GraphElementTypes.IsSolvedBy,
        date=topic_solver_node.date
    )


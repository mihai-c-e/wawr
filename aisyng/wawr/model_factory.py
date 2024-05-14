from aisyng.wawr.models import PaperAbstract, GraphElementTypes
from aisyng.base.models import GraphNode, GraphRelationship

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
        type_id=GraphElementTypes.Abstract,
    )

def create_title_node_from_abstract_info(abstract_node: GraphNode) -> GraphNode:
    node_info = abstract_node.meta
    return GraphNode(
        id=get_title_node_id(node_info),
        text=node_info.title,
        date=node_info.date,
        title=node_info.title,
        meta=node_info,
        type_id=GraphElementTypes.Title,
        source_id=node_info.id,
        text_type='title',
    )

def create_title_to_abstract_relationship(abstract_node: GraphNode, title_node: GraphNode) -> GraphRelationship:
    return GraphRelationship(
        id=get_relationship_id(title_node, GraphElementTypes.IsTitleOf, abstract_node),
        from_node=title_node,
        to_node=abstract_node,
        text="title of",
        type_id=GraphElementTypes.IsTitleOf,
        date=abstract_node.date,
    )
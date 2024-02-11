from __future__ import annotations

from datetime import datetime
from typing import List, Set, Dict, Any, Iterable, Optional
from uuid import uuid4

from pydantic import BaseModel
import numpy as np


class GraphElement(BaseModel):
    id: str
    created_date: datetime
    status: str = ""
    meta: Dict[str, Any] = dict()
    text: str
    date: Optional[datetime] = None
    type_id: str
    embeddings: Dict[str, List[float]] = dict()

    def __init__(self, **kwargs):
        super().__init__(
            type_id=kwargs.pop("type_id", type(self).__name__),
            id=kwargs.pop("id", str(uuid4())),
            created_date=kwargs.pop("created_date", datetime.now()),
            **kwargs
        )

    def __hash__(self):
        return self.id


class GraphNode(GraphElement):
    relationships: Set[GraphRelationship] = set()
    relationships_by_type: Dict[str, Set[GraphRelationship]] = dict()

    def __init__(self, text: str, **kwargs):
        super().__init__(text=text, **kwargs)

    def add_relationship(self, relationship: GraphRelationship):
        self.relationships.add(relationship)
        type_set = self.relationships_by_type.get(relationship.type_id, set())
        type_set.add(relationship)
        self.relationships_by_type[relationship.type_id] = type_set


class GraphRelationship(GraphElement):
    from_node: GraphNode = None
    from_node_id: str = None
    to_node: GraphNode = None
    to_node_id: str = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Graph:
    nodes: Dict[str, GraphNode] = dict()
    relationships: Set[GraphRelationship] = set()

    def add_all(self, nodes: List[GraphNode] = None, relationships: List[GraphRelationship] = None) -> None:
        nodes_to_add = {n.id: n for n in nodes}
        self.nodes.update(nodes_to_add)
        self.relationships.update(relationships)
        for rel in relationships:
            from_node = self.nodes.get(rel.from_node_id)
            to_node = self.nodes.get(rel.to_node_id)
            rel.from_node = from_node
            rel.to_node = to_node
            if from_node is not None:
                from_node.add_relationship(rel)
            if to_node is not None:
                to_node.add_relationship(rel)

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Set, Dict, Any, Iterable, Optional, Union
from uuid import uuid4

from pydantic import BaseModel
import numpy as np


class GraphElement(BaseModel):
    id: str
    created_date: datetime
    status: str = ""
    meta: Union[BaseModel, Dict] = dict()
    text: str
    date: Optional[datetime] = None
    type_id: str
    citation: Optional[str] = ""
    source_id: Optional[str] = None
    title: Optional[str] = None
    text_type: Optional[str] = None
    embeddings: Dict[str, List[float]] = dict()

    def __init__(self, **kwargs):
        super().__init__(
            type_id=kwargs.pop("type_id", type(self).__name__),
            id=kwargs.pop("id", str(uuid4())),
            created_date=kwargs.pop("created_date", datetime.now()),
            **kwargs
        )

    def __hash__(self):
        return self.id.__hash__()

    def meta_model_dump_json(self):
        return json.dumps(self.meta) if isinstance(self.meta, dict) else self.meta.model_dump_json()


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

    def get_from_node_id(self):
        return self.from_node.id if self.from_node is not None else self.from_node_id

    def get_to_node_id(self):
        return self.to_node.id if self.to_node is not None else self.to_node_id


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



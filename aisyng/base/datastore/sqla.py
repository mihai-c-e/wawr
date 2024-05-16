from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Callable, Any

from sqlalchemy import ForeignKey, Engine
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aisyng.base.embeddings import EmbeddingPool, Embedder
from aisyng.base.models import GraphElement, GraphNode, GraphRelationship, ScoredGraphElement
from aisyng.base.datastore.base import PersistenceInterface


class SQLABase(DeclarativeBase):
    pass


class SQLAElement(SQLABase):
    __tablename__ = "elements"

    record_type: Mapped[str]
    id: Mapped[str] = mapped_column(primary_key=True)
    type_id: Mapped[str]
    source_id: Mapped[str]
    meta: Mapped[dict] = mapped_column(JSON())
    text: Mapped[str]
    citation: Mapped[str]
    title: Mapped[str]
    text_type: Mapped[str]
    created_date: Mapped[datetime]
    date: Mapped[datetime] = mapped_column(nullable=True)
    status: Mapped[str]

    __mapper_args__ = {
        "polymorphic_identity": "node",
        "polymorphic_on": "record_type"
    }


class SQLARelationship(SQLAElement):
    __tablename__ = "relationships"
    id: Mapped[str] = mapped_column(ForeignKey("elements.id"), primary_key=True)
    from_node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"))
    to_node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"))

    __mapper_args__ = {
        "polymorphic_identity": "relationship",
        "inherit_condition": id == SQLAElement.id
    }


class SQLAPersistenceInterface(PersistenceInterface):
    embedding_pool: EmbeddingPool
    session_factory: Callable[[Any], Engine]

    def __init__(self, embedding_pool: EmbeddingPool, session_factory: Callable[[Any], Engine]):
        self.embedding_pool = embedding_pool
        self.session_factory = session_factory

    def find_by_similarity(
            self,
            with_strings: List[str],
            with_vectors: List[List[float]],
            distance_threshold: float,
            embedder: Embedder,
            limit: int,
            from_date: datetime = None,
            to_date: datetime = None,
            only_type_ids: List[str] = None,
            exclude_type_ids: List[str] = None, **kwargs
    ) -> List[ScoredGraphElement]:
        raise NotImplementedError

    def persist(
            self,
            objects_add: List[GraphElement] = None,
            objects_merge: List[GraphElement] = None,
            **kwargs
    ) -> None:
        if objects_add is None and objects_merge is None:
            raise ValueError("Object lists not provided")
        self.persist_sql_objects(
            objects_add=None if objects_add is None else self.elements_to_sql(objects_add),
            objects_merge=None if objects_merge is None else self.elements_to_sql(objects_merge),
        )

    def persist_embeddings(self, nodes: List[GraphNode], embedding_key: str, batch_size: int):
        embedder = self.embedding_pool.get_embedder(embedding_key)
        logging.info(f"Saving {len(nodes)} embeddings to database")
        for i in range(0, len(nodes), batch_size):
            sql_objects: List[embedder.table] = list()
            for node in nodes[i:i + batch_size]:
                emb_row = self.embedding_to_sql(node, embedding_key=embedder.name)
                sql_objects.append(emb_row)
            with self.session_factory() as sess:
                with sess.begin():
                    sess.add_all(sql_objects)

    def persist_sql_objects(self, objects_add: List[SQLAElement] = None, objects_merge: List[SQLAElement] = None,
                            **kwargs) -> bool:
        if objects_add is None and objects_merge is None:
            raise ValueError("Object lists not provided")
        with self.session_factory() as sess:
            with sess.begin():
                if objects_merge is not None:
                    for obj in objects_merge:
                        sess.merge(obj)
                if objects_add is not None:
                    objects_add.sort(key=lambda x: 1 if isinstance(x, SQLARelationship) else 0)
                    sess.add_all(objects_add)
        return True

    def _element_to_sql_object(self, element: GraphElement) -> SQLAElement:
        if isinstance(element, GraphNode):
            obj = SQLAElement()
            obj.record_type = "node"
        elif isinstance(element, GraphRelationship):
            obj = SQLARelationship()
            obj.record_type = "relationship"
            obj.from_node_id = element.from_node.id
            obj.to_node_id = element.to_node.id
        else:
            raise ValueError(f"Unknown type: {type(element)}")
        return obj

    def element_to_sql(self, element: GraphElement) -> SQLAElement:
        obj = self._element_to_sql_object(element)
        obj.id = element.id
        obj.source_id = element.source_id
        obj.meta = element.meta if isinstance(element.meta, dict) else element.meta.model_dump(mode='json')
        obj.text = element.text
        obj.date = element.date
        obj.created_date = element.created_date
        obj.status = element.status
        obj.type_id = type(element).__name__ if element.type_id is None else element.type_id
        obj.text_type = element.text_type
        obj.title = element.title
        obj.citation = element.citation

        return obj

    def elements_to_sql(self, elements: List[GraphElement]) -> List[SQLAElement]:
        sql_objects = list()
        for e in elements:
            sql_objects.append(self.element_to_sql(e))
            for embedding_key in e.embeddings.keys():
                sql_objects.append(self.embedding_to_sql(element=e, embedding_key=embedding_key))
        return sql_objects

    def sql_to_element(self, obj: SQLAElement) -> GraphElement:
        pass_kwargs = {
            "id": obj.id, "type_id": obj.type_id, "text": obj.text, "meta": obj.meta, "date": obj.date,
            "created_date": obj.created_date, "status": obj.status, "source_id": obj.source_id,
            "text_type": obj.text_type, "citation": obj.citation, "title": obj.title
        }
        if isinstance(obj, SQLARelationship):
            pass_kwargs.update({"from_node_id": obj.from_node_id, "to_node_id": obj.to_node_id})
            element = GraphRelationship.model_validate(pass_kwargs)
        elif isinstance(obj, SQLAElement):
            element = GraphNode.model_validate(pass_kwargs)
        else:
            raise ValueError(f"Unkonwn record type: {obj.record_type}")
        return element

    def sql_to_element_list(self, objects: List[SQLAElement]) -> List[GraphElement]:
        return [self.sql_to_element(obj) for obj in objects]

    def embedding_to_sql(self, element: GraphElement, embedding_key: str) -> SQLABase:
        embedder = self.embedding_pool.get_embedder(embedding_key)
        obj = embedder.table()
        obj.node_id = element.id
        obj.embedding = element.embeddings[embedder.name]
        return obj

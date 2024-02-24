from __future__ import annotations
import logging
import os
from typing import List, Type, Dict

from sqlalchemy import String, JSON, ForeignKey, create_engine, select, alias, Column, false, Numeric, Float, func, case
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, aliased, Query
from pgvector.sqlalchemy import Vector
from datetime import datetime

from ekb.base.models import GraphElement, GraphNode, GraphRelationship
from ekb.utils import get_subclass_by_name

_db_srv = os.environ["DATA_DB_SERVER"]
_db_usr = os.environ["DATA_DB_USR"]
_db_pwd = os.environ["DATA_DB_PWD"]
_db_port = os.environ["DATA_DB_PORT"]
_db = os.environ["DATA_DB"]
_con_str = f'postgresql+psycopg2://{_db_usr}:{_db_pwd}@{_db_srv}:{_db_port}/{_db}'
engine = create_engine(_con_str)
Session = sessionmaker(engine)


def create_all():
    SQLABase.metadata.create_all(engine)


class SQLABase(DeclarativeBase):
    pass

class SQLAElement(SQLABase):
    __tablename__ = "elements"

    record_type: Mapped[str]
    id: Mapped[str] = mapped_column(primary_key=True)
    type_id: Mapped[str]
    meta: Mapped[dict] = mapped_column(JSON())
    text: Mapped[str]
    created_date: Mapped[datetime]
    date: Mapped[datetime] = mapped_column(nullable=True)
    status: Mapped[str]
    __mapper_args__ = {
        "polymorphic_identity": "node",
        "polymorphic_on": "record_type"
    }
    @classmethod
    def find_by_similarity(
            cls,
            with_string: str,
            with_vector: List[float],
            distance_threshold: float,
            embedding_key: str,
            limit: int,
            from_date: datetime = None,
            to_date: datetime = None,
            type_ids: List[str] = None
    ) -> List[SQLAElement]:
        emb_table = get_embedding_table_class_by_key(embedding_key)
        subquery = select(
            SQLAElement,
            emb_table,
            case(
                (with_string is not None and with_string != '' and SQLAElement.text.ilike(with_string+'%'), 0.0),
                else_= emb_table.embedding.cosine_distance(with_vector)
            ).label("distance")
        ).select_from(SQLAElement).join(emb_table, onclause=SQLAElement.id==emb_table.node_id).subquery()
        g = aliased(SQLAElement, subquery)
        e = aliased(emb_table, subquery)
        query = Query(
            [g,
            subquery.columns["distance"]]
        ).select_from(subquery).where(
            (subquery.columns["distance"] <= distance_threshold)
            & (True if from_date is None else SQLAElement.date >= from_date)
            & (True if to_date is None else SQLAElement.date <= to_date)
            & (True if type_ids is None else SQLAElement.type_id.in_(type_ids))
        ).order_by(subquery.columns["distance"].asc()).limit(limit)
        with Session() as sess:
            results = [[r[0], r[1]] for r in sess.execute(query).all()]
            return results


class SQLARelationship(SQLAElement):
    __tablename__ = "relationships"
    id: Mapped[str] = mapped_column(ForeignKey("elements.id"), primary_key=True)
    from_node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"))
    to_node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"))

    __mapper_args__ = {
        "polymorphic_identity": "relationship",
        "inherit_condition": id == SQLAElement.id
    }

class OpenAITextEmbedding3Small(SQLABase):
    __tablename__ = "opneai_text_embedding_3_small"
    key_name: str = __tablename__
    node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"), primary_key=True)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))

__embedding_class_to_sql: Dict[str, Type] = {
        "text-embedding-3-small": OpenAITextEmbedding3Small
    }

def get_embedding_table_class_by_key(embedding_key: str) -> Type[SQLABase]:
    return __embedding_class_to_sql[embedding_key]

def _element_to_sql_object(element: GraphElement) -> SQLAElement:
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

def element_to_sql(element: GraphElement) -> SQLAElement:
    obj = _element_to_sql_object(element)
    obj.id = element.id
    obj.meta = element.meta
    obj.text = element.text
    obj.date = element.date
    obj.created_date = element.created_date
    obj.status = element.status
    obj.type_id = type(element).__name__
    return obj

def sql_to_element(obj: SQLAElement) -> GraphElement:
    #obj_class = get_subclass_by_name(GraphNode, obj.type_id)
    pass_kwargs = {
        "id": obj.id, "type_id": obj.type_id, "text": obj.text, "meta": obj.meta, "date": obj.date,
        "created_date": obj.created_date, "status": obj.status
    }
    if isinstance(obj, SQLAElement):
        element = GraphNode.model_validate(pass_kwargs)
    elif isinstance(obj, SQLARelationship):
        pass_kwargs.update({"from_node_id": obj.from_node_id, "to_node_id": obj.to_node_id})
        element = GraphRelationship.model_validate(pass_kwargs)
    else:
        raise ValueError(f"Unkonwn record type: {obj.record_type}")
    return element

def sql_to_element_list(objects: List[SQLAElement]) -> List[GraphElement]:
    return [sql_to_element(obj) for obj in objects]

def embedding_to_sql(element: GraphElement, embedding_key: str) -> SQLABase:
    obj = get_embedding_table_class_by_key(embedding_key)()
    obj.node_id = element.id
    obj.embedding = element.embeddings[embedding_key]
    return obj

def persist_graph_objects(objects: List[GraphElement], merge: bool = False) -> None:
    sql_objects = list()
    for obj in objects:
        sql_objects.append(element_to_sql(obj))
        for embedding_key in obj.embeddings.keys():
            sql_objects.append(embedding_to_sql(obj, embedding_key))
    persist_sql_objects(sql_objects, merge=merge)

def persist_sql_objects(objects: List[SQLABase], merge: bool = False) -> None:
    with Session() as sess:
        with sess.begin():
            if merge:
                for obj in objects:
                    sess.merge(obj)
            else:
                sess.add_all(objects)


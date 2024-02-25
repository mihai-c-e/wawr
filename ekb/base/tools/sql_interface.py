from __future__ import annotations
import logging
import os
from typing import List, Type, Dict, Tuple

from sqlalchemy import String, ForeignKey, create_engine, select, case
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, scoped_session, aliased, Query
from pgvector.sqlalchemy import Vector
from datetime import datetime

from ekb.base.topic import TopicNode
from ekb.base.models import GraphElement, GraphNode, GraphRelationship
from ekb.utils import get_subclass_by_name

_db_srv = os.environ["DATA_DB_SERVER"]
_db_usr = os.environ["DATA_DB_USR"]
_db_pwd = os.environ["DATA_DB_PWD"]
_db_port = os.environ["DATA_DB_PORT"]
_db = os.environ["DATA_DB"]
_con_str = f'postgresql+psycopg2://{_db_usr}:{_db_pwd}@{_db_srv}:{_db_port}/{_db}'
engine = create_engine(_con_str)
Session = scoped_session(sessionmaker(engine))


def create_all():
    SQLABase.metadata.create_all(engine)


class SQLABase(DeclarativeBase):
    pass

_registered_graph_element_types = {
    'TopicNode': TopicNode
}

def _get_class_for_element_type(element_type: str):
    return _registered_graph_element_types.get(element_type, None)

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
        subquery = (select(
            SQLAElement,
            emb_table,
            case(
                (with_string is not None and with_string != '' and SQLAElement.text.ilike(with_string+'%'), 0.0),
                else_= emb_table.embedding.cosine_distance(with_vector)
            ).label("distance")
        ).select_from(SQLAElement).join(emb_table, onclause=SQLAElement.id==emb_table.node_id)
        .where(
            (True if from_date is None else ((SQLAElement.date >= from_date) | (SQLAElement.date == None)))
            & (True if to_date is None else ((SQLAElement.date <= to_date) | (SQLAElement.date == None)))
            & (True if type_ids is None else SQLAElement.type_id.in_(type_ids))
        ).subquery())
        g = aliased(SQLAElement, subquery)
        e = aliased(emb_table, subquery)
        query = Query(
            [g,
            subquery.columns["distance"]]
        ).select_from(subquery).where(
            (subquery.columns["distance"] <= distance_threshold)

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

def elements_to_sql(elements: List[GraphElement]) -> List[SQLAElement]:
    sql_objects = list()
    for e in elements:
        sql_objects.append(element_to_sql(e))
        for embedding_key in e.embeddings.keys():
            sql_objects.append(embedding_to_sql(e, embedding_key))
    return sql_objects

def sql_to_element(obj: SQLAElement) -> GraphElement:
    #obj_class = get_subclass_by_name(GraphNode, obj.type_id)
    pass_kwargs = {
        "id": obj.id, "type_id": obj.type_id, "text": obj.text, "meta": obj.meta, "date": obj.date,
        "created_date": obj.created_date, "status": obj.status
    }
    clazz = _get_class_for_element_type(obj.type_id)
    if isinstance(obj, SQLARelationship):
        pass_kwargs.update({"from_node_id": obj.from_node_id, "to_node_id": obj.to_node_id})
        element = (clazz or GraphRelationship).model_validate(pass_kwargs)
    elif isinstance(obj, SQLAElement):
        element = (clazz or GraphNode).model_validate(pass_kwargs)
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

def persist_graph_elements(elements_add: List[GraphElement] = None, elements_merge: List[GraphElement] = None) -> None:
    if elements_add is None and elements_merge is None:
        raise ValueError("Object lists not provided")
    persist_sql_objects(
        objects_add = None if elements_add is None else elements_to_sql(elements_add),
        objects_merge=None if elements_merge is None else elements_to_sql(elements_merge),
    )

def persist_sql_objects(objects_add: List[SQLABase] = None, objects_merge: List[SQLABase] = None) -> None:
    if objects_add is None and objects_merge is None:
        raise ValueError("Object lists not provided")
    with Session() as sess:
        with sess.begin():
            if objects_merge is not None:
                for obj in objects_merge:
                    sess.merge(obj)
            if objects_add is not None:
                objects_add.sort(key=lambda x: 1 if isinstance(x, SQLARelationship) else 0)
                sess.add_all(objects_add)


def get_element_by_id(id: str) -> GraphElement | None:
    with Session() as sess:
        query = select(SQLAElement).where(SQLAElement.id == id)
        object = sess.execute(query).first()
    if object is None:
        return None
    element = sql_to_element(object[0])
    return element

def get_elements_by_ids(ids: List[str]) -> List[GraphElement]:
    if ids is None or len(ids) == 0:
        return list()
    query = select(SQLAElement).where(
        (SQLAElement.id.in_(ids))
    ).order_by(SQLAElement.date.desc())
    with Session() as sess:
        results_sql = [row[0] for row in sess.execute(query)]
        results = sql_to_element_list(results_sql)
    return results

def get_topic_node_and_subgraph(id: str) -> Tuple[GraphElement, List[GraphElement]] | None:
    topic_node = get_element_by_id(id)
    if topic_node is None:
        raise ValueError(f"Node id {id} not found")
    if not isinstance(topic_node, TopicNode):
        raise ValueError(f"Node id {id} is not a topic node")
    meta = topic_node.get_topic_meta()
    subgraph = get_elements_by_ids(meta.subgraph_ids)
    return topic_node, subgraph

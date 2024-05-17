import os
from datetime import datetime
from typing import List, Set

from sqlalchemy import create_engine, case, select, true
from sqlalchemy.orm import scoped_session, sessionmaker, aliased, Query

from aisyng.base.datastore.sqla import SQLAPersistenceInterface, SQLAElement
from aisyng.base.embeddings import EmbeddingPool, Embedder
from aisyng.base.models.graph import ScoredGraphElement
from aisyng.base.models.base import PayloadBase

_db_srv = os.environ["DATA_DB_SERVER"]
_db_usr = os.environ["DATA_DB_USR"]
_db_pwd = os.environ["DATA_DB_PWD"]
_db_port = os.environ["DATA_DB_PORT"]
_db = os.environ["DATA_DB"]
_con_str = f'postgresql+psycopg2://{_db_usr}:{_db_pwd}@{_db_srv}:{_db_port}/{_db}'
engine = create_engine(_con_str)
Session = scoped_session(sessionmaker(engine))


class PSQLPersistenceInterface(SQLAPersistenceInterface):
    def __init__(self, embedding_pool: EmbeddingPool, payload_types: Set[PayloadBase.__class__]):
        super().__init__(embedding_pool=embedding_pool, session_factory=Session, payload_types=payload_types)

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
            exclude_type_ids: List[str] = None,
            **kwargs
    ) -> List[ScoredGraphElement]:

        distance_formula = None
        for i in range(len(with_strings)):
            element = (1 - case(
                (with_strings[i] is not None and with_strings[i] != '' and SQLAElement.text.ilike(
                    '%' + with_strings[i] + '%'), 0.0),
                else_=embedder.table.embedding.cosine_distance(with_vectors[i])
            ))
            distance_formula = (element if distance_formula is None else (distance_formula * element))

        subquery = (select(
            SQLAElement,
            embedder.table,
            (1 - distance_formula).label("distance")
        ).select_from(SQLAElement).join(embedder.table, onclause=SQLAElement.id == embedder.table.node_id)
                    .where(
            (true() if from_date is None else ((SQLAElement.date >= from_date) | (SQLAElement.date == None)))
            & (true() if to_date is None else ((SQLAElement.date <= to_date) | (SQLAElement.date == None)))
            & (true() if only_type_ids is None else SQLAElement.type_id.in_(only_type_ids))
            & (true() if exclude_type_ids is None else SQLAElement.type_id.notin_(exclude_type_ids))
        ).subquery())
        g = aliased(SQLAElement, subquery)
        e = aliased(embedder.table, subquery)
        query = Query(
            [g,
             subquery.columns["distance"]]
        ).select_from(subquery).where(
            (subquery.columns["distance"] <= distance_threshold)

        ).order_by(subquery.columns["distance"].asc()).limit(limit)
        with Session() as sess:
            return [
                ScoredGraphElement(
                    element=self.sql_to_element(r[0]),
                    score=r[1]
                )
                for r in sess.execute(query).all()
            ]


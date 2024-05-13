import logging

from sqlalchemy import select

from aisyng.base.persistence.base import MultiMediaPersist
from aisyng.base.persistence.sqla import SQLAPersistenceInterface, SQLAElement
from aisyng.base.persistence.neo4j import Neo4JPersistenceInterface
from aisyng.wawr.models import GraphElementTypes


class WAWRPersistence(MultiMediaPersist):
    sqli: SQLAPersistenceInterface
    neo4ji: Neo4JPersistenceInterface

    def __init__(self, sqli: SQLAPersistenceInterface, neo4ji: Neo4JPersistenceInterface):
        super().__init__(media_list=[neo4ji, sqli])
        self.sqli = sqli
        self.neo4ji = neo4ji

    def get_last_ingested_abstract_id(self):
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).where(
                SQLAElement.type_id == GraphElementTypes.Abstract
            ).order_by(SQLAElement.id.desc()).limit(1)
            result = sess.execute(stmt).fetchone()
        logging.info(f"Last abstract id in database: {'none' if result is None else result[0].id}")
        return None if result is None else result[0].id
import logging
from typing import List

from sqlalchemy import select, Null

from aisyng.base.persistence.base import MultiMediaPersist
from aisyng.base.persistence.sqla import SQLAPersistenceInterface, SQLAElement, SQLARelationship
from aisyng.base.persistence.neo4j import Neo4JPersistenceInterface
from aisyng.wawr.models import GraphElementTypes, should_ignore_graph_element_duplicates
from aisyng.base.models import GraphNode, GraphElement


class WAWRPersistence(MultiMediaPersist):
    sqli: SQLAPersistenceInterface
    neo4ji: Neo4JPersistenceInterface

    def __init__(self, sqli: SQLAPersistenceInterface, neo4ji: Neo4JPersistenceInterface):
        super().__init__(media_list=[neo4ji, sqli])
        self.sqli = sqli
        self.neo4ji = neo4ji

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                merge_ignorable_duplicates: bool = True, **kwargs) -> bool:
        if merge_ignorable_duplicates:
            objects_add_new = list()
            objects_merge_new = list()
            for obj in objects_add:
                if should_ignore_graph_element_duplicates(obj):
                    objects_merge_new.append(obj)
                else:
                    objects_add_new.append(obj)
            if len(objects_merge_new) > 0:
                if objects_merge is None:
                    objects_merge = list()
                objects_merge_new.extend(objects_merge)
            return super().persist(objects_add=objects_add_new, objects_merge=objects_merge_new, **kwargs)
        else:
            return super().persist(objects_add=objects_add, objects_merge=objects_merge)

    def get_last_ingested_abstract_id(self) -> str:
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).where(
                SQLAElement.type_id == GraphElementTypes.Abstract
            ).order_by(SQLAElement.id.desc()).limit(1)
            result = sess.execute(stmt).fetchone()
        logging.info(f"Last abstract id in database: {'none' if result is None else result[0].id}")
        return None if result is None else result[0].id

    def get_abstracts_without_facts(self, limit: int = 10) -> List[GraphNode]:
        rels = select(SQLARelationship).where(
                    SQLARelationship.type_id == GraphElementTypes.IsExtractedFrom
                ).subquery()
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).join(
                rels,
                onclause=SQLAElement.id==rels.c.to_node_id, isouter=True
            ).where(
                (SQLAElement.type_id == GraphElementTypes.Abstract) &
                (rels.c.to_node_id == None)
            ).order_by(SQLAElement.date.desc()).limit(limit)
            result = sess.execute(stmt).all()
            nodes = self.sqli.sql_to_element_list([n[0] for n in result])
        if len(nodes) > 0:
            logging.info(f"Loaded {len(nodes)} abstracts between {nodes[-1].date} and {nodes[0].date}")
        return nodes
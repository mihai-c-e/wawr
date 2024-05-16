import logging
from typing import List, cast

from sqlalchemy import select, Null
from sqlalchemy.orm import aliased

from aisyng.base.datastore.base import MultiMediaPersist
from aisyng.base.datastore.sqla import SQLAPersistenceInterface, SQLAElement, SQLARelationship
from aisyng.base.datastore.neo4j import Neo4JPersistenceInterface
from aisyng.wawr.models import GraphElementTypes, should_ignore_graph_element_duplicates
from aisyng.base.models import GraphNode, GraphElement, GraphRelationship


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

    def get_parents_without_extracted_children(
            self,
            parent_type_id: GraphElementTypes,
            limit: int = 10) -> List[GraphNode]:
        rels = select(SQLARelationship).where(
                    SQLARelationship.type_id == GraphElementTypes.IsExtractedFrom
                ).subquery()
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).join(
                rels,
                onclause=SQLAElement.id==rels.c.to_node_id, isouter=True
            ).where(
                (SQLAElement.type_id == parent_type_id) &
                (rels.c.to_node_id == None)
            ).order_by(SQLAElement.date.desc()).limit(limit)
            result = sess.execute(stmt).all()
            nodes = self.sqli.sql_to_element_list([n[0] for n in result])
        if len(nodes) > 0:
            logging.info(f"Loaded {len(nodes)} elements between {nodes[-1].date} and {nodes[0].date}")
        return cast(List[GraphNode], nodes)

    def get_node_by_id(self, id: str) -> GraphNode:
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).where(
                (SQLAElement.id == id)
            )
            result = sess.execute(stmt).first()
            result_list = list(result)
            if len(result_list) == 0:
                return None
            node = self.sqli.sql_to_element(result_list[0])
        return cast(GraphNode, node)

    def get_all_facts_and_fact_types(self, limit: int) -> List[GraphNode]:
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).where(
                SQLAElement.type_id.in_([GraphElementTypes.Fact, GraphElementTypes.FactType])
            ).order_by(SQLAElement.date.desc()).limit(limit)
            result = sess.execute(stmt).all()
            nodes = self.sqli.sql_to_element_list([n[0] for n in result])
        return cast(List[GraphNode], nodes)

    def get_all_fact_and_fact_type_relationships(self, limit: int) -> List[GraphRelationship]:
        from_node = aliased(SQLAElement)
        with self.sqli.session_factory() as sess:

            stmt = select(SQLARelationship).join(
                    from_node,
                    onclause=SQLARelationship.from_node_id==from_node.id
                ).where(
                (from_node.type_id==GraphElementTypes.Fact) &
                SQLARelationship.type_id.in_([GraphElementTypes.IsExtractedFrom, GraphElementTypes.IsA])
            ).order_by(SQLAElement.date.desc()).limit(limit)
            result = sess.execute(stmt).all()
            nodes = self.sqli.sql_to_element_list([n[0] for n in result])
        return cast(List[GraphRelationship], nodes)

    def get_nodes_without_embeddings(self, embedding_key: str, limit: int) -> List[GraphNode]:
        embedder = self.sqli.embedding_pool.get_embedder(embedding_key)
        embeddings_table = embedder.table
        with self.sqli.session_factory() as sess:
            stmt = select(SQLAElement).join(
                embeddings_table, isouter=True, onclause=(SQLAElement.id == embeddings_table.node_id)
            ).where(
                (embeddings_table.node_id == None) & (SQLAElement.record_type == 'node')
            ).limit(limit)
            result = sess.execute(stmt).all()
            nodes = [self.sqli.sql_to_element(n[0]) for n in result]
        if len(nodes) > 0:
            logging.info(f"Loaded {len(nodes)} nodes without embeddings")
        return cast(List[GraphNode], nodes)

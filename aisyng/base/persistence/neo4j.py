import logging
import os
from typing import List
from neo4j import GraphDatabase, Transaction, ManagedTransaction

from aisyng.base.persistence.base import PersistenceInterface
from aisyng.base.models import GraphElement, GraphNode, GraphRelationship

_connection_uri = os.environ["AURA_CONNECTION_URI"]
_user = os.environ["AURA_USERNAME"]
_password = os.environ["AURA_PASSWORD"]
_db = os.environ["AURA_DB"]
_auth = (_user, _password)


class Neo4JPersistenceInterface(PersistenceInterface):

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        with GraphDatabase.driver(_connection_uri, auth=_auth) as driver:
            with driver.session(database=_db) as session:
                session.execute_write(self._create_key_constraints_tx, objects_merge+objects_add)
                for obj in objects_add:
                    session.execute_write(self._create_graph_element_tx, obj, False)
        return True

    def _create_node_unique_constraint_query(self, node_type: str) -> str:
        return (f"CREATE CONSTRAINT node_type_{node_type}_key IF NOT EXISTS "
                f"FOR (node:{node_type}) REQUIRE (node.id) IS NODE KEY")

    def _create_relationship_unique_constraint_query(self, rel_type: str) -> str:
        return (f"CREATE CONSTRAINT rel_type_{rel_type}_key IF NOT_EXISTS "
                f"FOR ()-[rel:{rel_type}] REQUIRE (rel.id) IS RELATIONSHIP KEY")

    def _create_key_constraints_tx(self, tx, objects: List[GraphElement]) -> None:
        node_types = {o.type_id for o in objects if isinstance(o, GraphNode)}
        rel_types = {o.type_id for o in objects if isinstance(o, GraphRelationship)}
        logging.info(f"Found {len(node_types)} node types and {len(rel_types)} relationship types")
        logging.info(f"Creating missing key constraints")
        constraints = ([self._create_node_unique_constraint_query(t) for t in node_types] +
                [self._create_relationship_unique_constraint_query(t) for t in rel_types])
        for c in constraints:
            tx.run(c)
        logging.info(f"Key constraints created")


    def _create_graph_element_tx(self, tx: ManagedTransaction, element: GraphElement, merge: bool):
        if isinstance(element, GraphNode):
            return self._create_node_tx(tx, element, merge)
        elif isinstance(element, GraphRelationship):
            return self._create_relationship_tx(tx, element, merge)

    def _create_node_tx(self, tx: ManagedTransaction, element: GraphNode, merge: bool):
        result = tx.run(f"MERGE (n: {element.type_id}:ALL {{"
                        "id: $id, created_date: $created_date, status: $status, text: $text, date: $date, "
                        "type_id: $type_id, citation: $citation, source_id: $source_id, title: $title, "
                        "text_type: $text_type, meta: $meta"
                        "})",
                        id=element.id,
                        created_date=element.created_date,
                        status=element.status,
                        text=element.text,
                        date=element.date,
                        type_id=element.type_id,
                        citation=element.citation,
                        source_id=element.source_id or "",
                        title=element.title,
                        text_type=element.text_type or "",
                        meta=element.meta_model_dump_json()
                        )
        return result

    def _create_relationship_tx(self, tx: ManagedTransaction, element: GraphRelationship, merge: bool):
        result = tx.run("MATCH (n1 {id: $from_node_id})"
                        "MATCH (n2 {id: $to_node_id})"
                        f"MERGE (n1)-[r: {element.type_id} {{"
                        "id: $id, created_date: $created_date, status: $status, text: $text, date: $date, "
                        "type_id: $type_id, citation: $citation, source_id: $source_id, title: $title, "
                        "text_type: $text_type, meta: $meta"
                        "}]->(n2)",
                        id=element.id,
                        created_date=element.created_date,
                        status=element.status,
                        text=element.text,
                        date=element.date,
                        type_id=element.type_id,
                        citation=element.citation,
                        source_id=element.source_id or "",
                        title=element.title,
                        text_type=element.text_type or "",
                        meta=element.meta_model_dump_json()
                        )

        return result
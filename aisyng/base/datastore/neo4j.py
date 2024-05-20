from __future__ import annotations

import json
import logging
import os
from typing import List, Any, Set, Dict
from neo4j import GraphDatabase, ManagedTransaction
from neo4j.graph import Node, Relationship, Path, Entity

from aisyng.base.datastore.base import PersistenceInterface
from aisyng.base.models.graph import GraphElement, GraphNode, GraphRelationship
from aisyng.base.models.base import PayloadBase

_connection_uri = os.environ["AURA_CONNECTION_URI"]
_user = os.environ["AURA_USERNAME"]
_password = os.environ["AURA_PASSWORD"]
_db = os.environ["AURA_DB"]
_auth = (_user, _password)


class Neo4JPersistenceInterface(PersistenceInterface):

    def __init__(self, payload_types: Set[PayloadBase.__class__]):
        self.payload_types = payload_types

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        all_objects = ((objects_add if isinstance(objects_add, List) else list()) +
                       (objects_merge if isinstance(objects_merge, List) else list()))
        all_objects.sort(key=lambda x: 1 if isinstance(x, GraphRelationship) else 0)
        with GraphDatabase.driver(_connection_uri, auth=_auth) as driver:
            with driver.session(database=_db) as session:
                session.execute_write(self._create_key_constraints_tx, all_objects)
                for obj in all_objects:
                    session.execute_write(self._create_graph_element_tx, obj, False)
        return True

    def _create_node_unique_constraint_query(self, node_type: str) -> str:
        return (f"CREATE CONSTRAINT node_type_{node_type}_key IF NOT EXISTS "
                f"FOR (node:{node_type}) REQUIRE (node.id) IS NODE KEY")

    def _create_relationship_unique_constraint_query(self, rel_type: str) -> str:
        return (f"CREATE CONSTRAINT rel_type_{rel_type}_key IF NOT EXISTS "
                f"FOR ()-[rel:{rel_type}]-() REQUIRE (rel.id) IS RELATIONSHIP KEY")

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
        result = tx.run(f"MERGE (n:{element.type_id} {{id: $id}}) SET "
                        "n.`id`=$id, n.`created_date`=$created_date, n.`status`=$status, n.`text`=$text, "
                        "n.`type_id`=$type_id, n.`citation`=$citation, n.`source_id`=$source_id, "
                        "n.`text_type`=$text_type, n.`meta`=$meta"
                        f"{', n.`date`=$date' if element.date is not None else ''}"
                        f"{', n.`title`=$title' if element.date is not None else ''}"
                        " RETURN n.id",
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
                        meta="{}" if element.meta is None else json.dumps(element.meta) if isinstance(
                            element.meta, dict) else element.meta_model_dump_json()
                        )

        return list(result)

    def _create_relationship_tx(self, tx: ManagedTransaction, element: GraphRelationship, merge: bool):
        result = tx.run("MATCH (n1 {id: $from_node_id}) "
                        "MATCH (n2 {id: $to_node_id}) "
                        f"MERGE (n1)-[r: {element.type_id} {{"
                        "id: $id, created_date: $created_date, status: $status, text: $text, "
                        "type_id: $type_id, citation: $citation, source_id: $source_id, "
                        "text_type: $text_type, meta: $meta"
                        f"{', date: $date' if element.date is not None else ''}"
                        f"{', title: $title' if element.date is not None else ''}"
                        "}]->(n2)",
                        from_node_id=element.get_from_node_id(),
                        to_node_id=element.get_to_node_id(),
                        id=element.id,
                        created_date=element.created_date,
                        status=element.status,
                        text=element.text,
                        date=element.date or "",
                        type_id=element.type_id,
                        citation=element.citation or "",
                        source_id=element.source_id or "",
                        title=element.title or "",
                        text_type=element.text_type or "",
                        meta=element.meta_model_dump_json()
                        )

        return result


    def get_paths_between(
            self,
            from_node_ids: List[str],
            to_node_labels: List[str],
            via_relationships: List[str],
            max_hops: int = -1,
            **kwargs
    ) -> Any:
        ids_qp = _list_to_cypher_list(from_node_ids)
        via_relationships_qp = _list_to_str(via_relationships, '|')
        to_node_labels_qp = _list_to_str(to_node_labels, '|')
        hops_query_part = f"*1..{max_hops}" if max_hops > 0 else ""
        query = (f"MATCH path=(n1)-[r:{via_relationships_qp}{hops_query_part}]-(n2:{to_node_labels_qp}) "
                 f"WHERE n1.id in {ids_qp} RETURN path")
        with GraphDatabase.driver(_connection_uri, auth=_auth) as driver:
            result = driver.execute_query(query)
        paths = [self.neo4j_path_to_graph_elements(record[0]) for record in result[0]]
        return paths

    def neo4j_path_to_graph_elements(self, path: Path) -> List[GraphElement]:
        return ([self.neo4j_node_to_graph_node(node) for node in path.nodes] +
                [self.neo4j_rel_to_graph_rel(rel) for rel in path.relationships])

    def _properties_to_dict(self, node_or_rel: Entity) -> Dict[str, Any]:
        properties = dict(node_or_rel.items())
        #TODO Temporary fix, ensure that meta is always part of the data model
        if not "meta" in properties:
            properties["meta"] = dict()
        else:
            properties["meta"] = json.loads(properties["meta"]) or dict()
        if not isinstance(properties, dict):
            raise ValueError(f"Expected dict, got {type(properties['meta'])} when loading graph node")
        properties["meta"]["type_id"] = properties.get("type_id")
        return properties

    def neo4j_node_to_graph_node(self, node: Node) -> GraphNode:
        properties = self._properties_to_dict(node)
        properties["meta"] = self.create_payload_object_from_graph_dict(properties)
        return GraphNode(**properties)

    def neo4j_rel_to_graph_rel(self, rel: Relationship) -> GraphRelationship:
        properties = self._properties_to_dict(rel)
        properties["from_node_id"] = dict(rel.start_node)["id"]
        properties["to_node_id"] = dict(rel.end_node)["id"]
        properties["meta"] = self.create_payload_object_from_graph_dict(properties)
        return GraphRelationship(**properties)


def _list_to_str(input_list: List[Any] | str, separator: str) -> str:
    if isinstance(input_list, str):
        return input_list
    if len(input_list) == 1:
        return input_list[0]
    return separator.join(input_list)

def _list_to_cypher_list(input_list: List[Any]) -> str:
    list_of_strings = [f"'{e}'" for e in input_list]
    return f"[{_list_to_str(input_list=list_of_strings, separator=',')}]"
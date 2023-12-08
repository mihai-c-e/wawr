from neo4j import GraphDatabase
from py2neo.data import Node, Path
from py2neo import Graph
import os
from wawr.utils import check_env
from typing import List, Set, Dict, Any
import logging

class Neo4jConnection:

    def __init__(self):
        self.graph = None
        self.connect()

    def check_env(self):
        required_keys = {'AURA_CONNECTION_URI', 'AURA_USERNAME', 'AURA_PASSWORD'}
        check_env(required_keys)

    def connect(self):
        self.check_env()
        n4jdriver = GraphDatabase.driver(
            os.environ['AURA_CONNECTION_URI'],
            auth=(os.environ['AURA_USERNAME'], os.environ['AURA_PASSWORD'])
        )
        graph = Graph(os.environ['AURA_CONNECTION_URI'],
                      auth=(os.environ['AURA_USERNAME'], os.environ['AURA_PASSWORD'])
                      )
        self.graph = graph
        return graph

    def get_top_nodes(self, limit=None):
        query = (f"MATCH (n)-[r]-(m) RETURN n, count(r) ORDER BY count(r) DESC "
                 f"{('LIMIT '+limit) if limit else ''};")
        cursor = self.graph.run(query)
        to_ret = list()
        while cursor.forward():
            to_ret.append(cursor.current)
        return to_ret

    def get_all_nodes(self):
        query = (f"MATCH (n) RETURN n")
        cursor = self.graph.run(query)
        to_ret = list()
        while cursor.forward():
            to_ret.append(cursor.current)
        return to_ret

    def get_all_node_keywords(self) -> Dict[str, Set]:
        query = (f"MATCH (n) RETURN labels(n), properties(n)")
        cursor = self.graph.run(query)
        to_ret = {'labels': set(), 'keys': set(), 'values': set()}
        while cursor.forward():
            to_ret['labels'].update(cursor.current[0])
            to_ret['keys'].update(cursor.current[1].keys())
            to_ret['values'].update(cursor.current[1].values())
        return to_ret


    def get_all_relation_keywords(self) -> List[str]:
        query = (f"CALL db.relationshipTypes()")
        cursor = self.graph.run(query)
        to_ret = list()
        while cursor.forward():
            to_ret.append(cursor.current[0])
        return to_ret

    def query_for_paths(self, node_set: Set[str], relationships: str, depth: str = '*') -> List[Path]:
        node_set = '["'+ '","'.join([n for n in node_set]) + '"]'
        query = f"""
            MATCH paths=shortestpath((n1)-[r{depth}]-(n2)) 
            WHERE (labels(n1) IN {node_set} OR n1.name IN {node_set}) 
                AND (labels(n2) IN {node_set} OR n2.name IN {node_set}) 
                AND n1.name <> n2.name
                AND reduce(total=0, r in relationships(paths)|case when type(r) in {relationships} then 1 else 0 end) > 0
            RETURN paths LIMIT 150
            
        """
        """
        UNION
            MATCH paths=((n3)-[r*1]-(n4)) 
            WHERE (labels(n3) IN {node_set} OR n3.name IN {node_set})         
            RETURN paths LIMIT 50
        """
        cursor = self.graph.run(query)
        to_ret = [r[0] for r in list(cursor)]
        return to_ret





if __name__ == '__main__':
    con = Neo4jConnection()
    con.connect()
    con.get_all_node_strings()
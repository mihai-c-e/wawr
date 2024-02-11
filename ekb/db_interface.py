from neo4j import GraphDatabase
from py2neo.data import Node, Path
from py2neo import Graph
import os
from ekb.utils import check_env, read_html
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

    def query_for_paths_v3(self, nodes: List[List[str]], relationships: str, depth: str = '*', limit: int = 100) -> (
            List)[Path]:

        # WITH [...] AS list_0, [...] AS list 1 ...
        list_as_query = lambda x: '["' + '","'.join([p for p in x]) + '"]'
        list_vars = []
        list_exprs = []
        for i, nodes_list in enumerate(nodes):
            var_name = f'list_{i}'
            list_vars.append(var_name)
            list_expr = list_as_query(nodes_list)
            list_exprs.append(list_expr)
        query = 'WITH ' + ','.join([f'{list_exprs[i]} AS {list_vars[i]}' for i in range(len(list_vars))]) + ' '

        # MATCH...
        query += f'MATCH p=shortestpath((n1)-[r{depth}]->(n2)) WHERE '
        conditions = []
        for var in list_vars:
            second_list = list(list_vars)
            second_list.remove(var)
            second_list_query = '(' + '+'.join(second_list) + ')'
            condition = f'n1.name in {var} and n2.name in {second_list_query}'
            conditions.append(condition)
        conditions_query = '(' + ') OR ('.join(conditions) + ') '
        query += conditions_query
        query += f' RETURN p ORDER BY length(p) ASC LIMIT {limit}'

        cursor = self.graph.run(query)
        as_list = list(cursor)
        if len(as_list) > 0 and len(as_list[0]) == 0:
            raise ValueError('Expected a list of N records X 1 item per recorc')
        to_ret = [r[0] for r in as_list]
        return to_ret


if __name__ == '__main__':
    con = Neo4jConnection()
    con.connect()
    con.get_all_node_strings()
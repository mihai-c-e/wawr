from wawr.llm_interface import ChatGPTConnection, EmbeddingService
from wawr.matcher import KeywordMatchingService
from wawr.db_interface import Neo4jConnection
from py2neo.data import Node, Relationship, Path
from py2neo import walk
from typing import List, Set, Dict, Any
import logging
import itertools
from typing import Iterable
import threading
import time

class QAResponse:
    def __init__(self, id, question, observer = None):
        self.id = id
        self.lock = threading.RLock()
        self.question = question
        self.keywords = None
        self.node_keywords_match_raw = None
        self.relationship_keywords_match_raw = None
        self.node_keywords_match = None
        self.relationship_keywords_match = None
        self.paths = None
        self.refs = None
        self.refs_list = None
        self.retrieved_nodes = None
        self.retrieved_relationships = None
        self.answer = None
        self.progress = 0.0
        self.status = "Initialised"
        self.observers = list(observer) if isinstance(observer, Iterable) \
            else [observer,] if observer is not None else list()

    def synchronised_update_value(self, member, value):
        if member not in self.__dict__:
            raise ValueError(f'{member} is not a member of this class')
        with self.lock:
            self.__dict__[member] = value
        self.updated(member, value)

    def synchronised_get_value(self, member):
        if member not in self.__dict__:
            raise ValueError(f'{member} is not a member of this class')
        with self.lock:
            return self.__dict__[member]

    def updated(self, member, value):
        for o in self.observers:
            o.updated(self, member, value)




class QAAgent:
    def __init__(self, query_depth: int=2, style: str = 'journalistic', temperature: float = 0.1, lazy: bool = True):
        self.lazy = lazy
        self.style = style
        self.temperature = temperature
        self.query_depth = query_depth
        if not lazy:
            self._init_connections()
        else:
            self.llm = None
            self.db = None
            self.embedding_service = None
            self.kw_matching_service = None

    def _init_connections(self):
        self.llm = ChatGPTConnection(model_name='gpt-4-1106-preview')
        self.db = Neo4jConnection()
        self.embedding_service = EmbeddingService()
        self.kw_matching_service = KeywordMatchingService()

    def threaded_answer(self, q: str, observers = None, id=None, sleep_first: int = 10):
        response = QAResponse(question=q, observer=observers, id=id)
        thread = threading.Thread(target = self.answer, args=(q, response))
        thread.start()
        return response, thread

    def _extract_unique_nodes(self, paths: List[Path]) -> List[Node]:
        return list(set([n for path in paths for n in path.nodes]))

    def _extract_unique_relationships(self, paths: List[Path]) -> List[Relationship]:
        return list(set([r for path in paths for r in path.relationships]))

    def _extract_reference_relationships(self, paths: List[Path]) -> List[Relationship]:
        return list(set([r for path in paths for r in path.relationships if 'summary' in r]))

    def _assign_relationship_id(self, relationships: List[Relationship]) -> None:
        for i, rel in enumerate(relationships):
            rel['reference_id'] = i + 1

    def refs_to_text(self, refs_list: List[Relationship]) -> str:
        l = lambda ref: f"{ref['reference_id']}. {ref['research']} from {ref['title']}: \"{ref['summary']}\""
        return '\n'.join([l(rel) for rel in refs_list])


    def answer(self, q: str, response = None, observers = None, id = None, sleep_first: int=5) -> QAResponse:
        if sleep_first:
            time.sleep(sleep_first)
        response.synchronised_update_value("progress", 0.0)
        if self.lazy:
            response.synchronised_update_value("status", "Connecting...")
            self._init_connections()

        if not response:
            response = QAResponse(question=q, observer=observers, id =id)
        response.synchronised_update_value("status", "Mapping question to Cypher query...")
        response.synchronised_update_value("progress", 0.2)
        keywords = self.llm.question_to_keywords(q)
        keywords['relations'].extend(['is', 'part of'])
        response.synchronised_update_value("keywords", keywords)
        response.synchronised_update_value("progress", 0.25)
        response.synchronised_update_value("node_keywords_match_raw", self.kw_matching_service.match_keywords_node(keywords['nodes'], k=50))
        response.synchronised_update_value("progress", 0.3)
        response.synchronised_update_value("relationship_keywords_match_raw", self.kw_matching_service.match_keywords_relationship(keywords['relations'], k=50))
        response.synchronised_update_value("progress", 0.35)

        nodes = list(itertools.chain(
            *[[q[0].page_content for q in response.synchronised_get_value("node_keywords_match_raw")[k]]
                for k in response.synchronised_get_value("node_keywords_match_raw").keys()]
        ))
        response.synchronised_update_value("node_keywords_match", nodes)
        response.synchronised_update_value("progress", 0.4)
        relationships = set()
        for k in response.synchronised_get_value("relationship_keywords_match_raw").keys():
            relationships.update({v[0].page_content for v in response.synchronised_get_value("relationship_keywords_match_raw")[k]})
        relationships = list(relationships)
        response.synchronised_update_value("relationship_keywords_match", relationships)
        response.synchronised_update_value("progress", 0.45)
        #nodes1 = keywords_match['nodes'][0]
        #nodes2 = keywords_match['nodes'][1]
        #graph_results, refs = self.db.query_for_paths_and_relationships(nodes1, nodes2, relationships, depth='*1..5')

        response.synchronised_update_value("status", "Querying knowledge graph...")
        depth = f'*0..{self.query_depth}'
        paths = self.db.query_for_paths(nodes, relationships, depth=depth)
        response.synchronised_update_value("paths", paths)

        response.synchronised_update_value("retrieved_nodes", self._extract_unique_nodes(paths))

        unique_relationships = self._extract_unique_relationships(paths)
        response.synchronised_update_value("retrieved_relationships", unique_relationships)

        refs_list = self._extract_reference_relationships(paths)
        self._assign_relationship_id(refs_list)
        response.synchronised_update_value("refs_list", refs_list)

        refs = self.refs_to_text(refs_list)
        response.synchronised_update_value("refs", refs)

        response.synchronised_update_value("progress", 0.7)

        response.synchronised_update_value("status", "Composing answer...")
        response.synchronised_update_value("answer", self.llm.compose_answer_from_paths_news_style(q, refs, refs, style=self.style, temperature=self.temperature))
        response.synchronised_update_value("status", "Completed")
        response.synchronised_update_value("progress", 1.0)
        return response


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    class QAResponseObserver:
        def updated(self, bj, member, value):
            print(f'Field {member} updated with {value}')

    agent = QAAgent(query_depth=2)
    q = 'What are some applications of large language models in medical science?'
    #q = 'Which opinion on gpt-4 is the most divergent?'
    response, thread = agent.threaded_answer(
        q, observers=QAResponseObserver()
    )
    thread.join()
    print('Done')
    #agent.answer(' ?')
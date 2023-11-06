from wawr.llm_interface import ChatGPTConnection, EmbeddingService
from wawr.matcher import KeywordMatchingService
from wawr.db_interface import Neo4jConnection
import logging
import itertools
from typing import Iterable
import threading

class QAResponse:
    def __init__(self, question, observer = None):
        self.lock = threading.RLock()
        self.question = question
        self.keywords = None
        self.node_keywords_match_raw = None
        self.relationship_keywords_match_raw = None
        self.node_keywords_match = None
        self.relationship_keywords_match = None
        self.graph_results = None
        self.refs = None
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
    def __init__(self):
        self.llm = ChatGPTConnection()
        self.db = Neo4jConnection()
        self.embedding_service = EmbeddingService()
        self.kw_matching_service = KeywordMatchingService()

    def async_answer(self, q: str, observers = None):
        response = QAResponse(q, observers)
        thread = threading.Thread(target = self.answer, args=(q, response))
        thread.start()
        return response, thread

    def answer(self, q: str, response = None, observers = None) -> QAResponse:
        if not response:
            response = QAResponse(q, observers)
        response.synchronised_update_value("status", "Getting keyword recommendations...")
        keywords = self.llm.question_to_keywords(q)
        keywords['relations'].extend(['is', 'part of'])
        response.synchronised_update_value("keywords", keywords)

        response.synchronised_update_value("status", "Matching keyword recommendations with actual values...")
        response.synchronised_update_value("node_keywords_match_raw", self.kw_matching_service.match_keywords_node(keywords['nodes'], k=50))
        response.synchronised_update_value("relationship_keywords_match_raw", self.kw_matching_service.match_keywords_relationship(keywords['relations'], k=50))
        #q_match = self.kw_matching_service.match_keywords([q,], k=100)

        nodes = list(itertools.chain(
            *[[q[0].page_content for q in response.synchronised_get_value("node_keywords_match_raw")[k]]
                for k in response.synchronised_get_value("node_keywords_match_raw").keys()]
        ))
        response.synchronised_update_value("node_keywords_match", nodes)

        relationships = set()
        for k in response.synchronised_get_value("relationship_keywords_match_raw").keys():
            relationships.update({v[0].page_content for v in response.synchronised_get_value("relationship_keywords_match_raw")[k]})
        relationships = list(relationships)
        response.synchronised_update_value("relationship_keywords_match", relationships)
        #nodes1 = keywords_match['nodes'][0]
        #nodes2 = keywords_match['nodes'][1]
        #graph_results, refs = self.db.query_for_paths_and_relationships(nodes1, nodes2, relationships, depth='*1..5')

        response.synchronised_update_value("status", "Querying knowledge graph...")
        graph_results, refs, nodes, relationships = self.db.query_for_paths_and_relationships2(nodes, relationships, depth='*0..2')
        response.synchronised_update_value("graph_results", graph_results)
        response.synchronised_update_value("refs", refs)
        response.synchronised_update_value("retrieved_nodes", nodes)
        response.synchronised_update_value("retrieved_relationships", relationships)

        response.synchronised_update_value("status", "Composing answer...")
        response.synchronised_update_value("answer", self.llm.compose_answer_from_paths_news_style(q, graph_results, refs))
        response.synchronised_update_value("status", "Completed")
        return response


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)


    class QAResponseObserver:
        def updated(self, bj, member, value):
            print(f'Field {member} updated with {value}')

    agent = QAAgent()
    response, thread = agent.async_answer(
        'What are some applications of large language models in medical science?', observers=QAResponseObserver()
    )
    thread.join()
    #agent.answer('What improvements have been brought to large language models?')
    #agent.answer('What do you know about GPT-4 and its applications?')
    #agent.answer('What do you know about stock market prediction with GPT models?')
    #agent.answer('What is GPT-4 and what are its applications?')
    #agent.answer('Can large language models be used for code generation?')
    #agent.answer('What do you know about knowledge graphs?')
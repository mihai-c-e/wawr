import pandas as pd
from wawr.llm_interface import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import VectorStore, SKLearnVectorStore
from typing import List
import os
import threading

import logging

class KeywordMatchingService:
    def __init__(self, data_folder: str = None):
        self.embedder = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API'])
        self.lock = threading.RLock()
        data_folder = data_folder or os.environ.get('DATA_FOLDER')
        if not data_folder:
            raise ValueError('data_folder missing from constructor arguments and DATA_FOLDER missing from env')
        kw_persist_path = os.path.join(data_folder, 'sklearn_node_keywords')
        self.vector_store_node = SKLearnVectorStore(
            embedding=self.embedder, persist_path=kw_persist_path
        )

        rel_persist_path = os.path.join(data_folder, 'sklearn_relationship_keywords')
        self.vector_store_relationship = SKLearnVectorStore(
            embedding=self.embedder, persist_path=rel_persist_path
        )

    def __match_keywords(self, kw_list: List[str], k: int, vector_store: VectorStore, threshold: float = 0.2,
                         takes: int = 1):
        with self.lock:
            if takes <= 0:
                raise ValueError("Number of takes cannot be negative")
            i = 0
            to_ret = None
            while i < takes:
                if i > 0 and to_ret:
                    kw_list = [
                        ', '.join([
                            k[0].page_content for j, k in enumerate(v) if j < i + 1 and k[1] <= threshold
                        ])
                        for v in to_ret.values()
                    ]
                to_ret = {kw: [r for r in vector_store.similarity_search_with_score(kw, k=k) if r[1] < threshold] for kw
                          in kw_list}
                i += 1
        return to_ret

    def match_keywords_node(self, kw_list: List[str], k: int, threshold: float = 0.2, takes: int = 1):
        return self.__match_keywords(kw_list, k, self.vector_store_node, threshold=threshold, takes=takes)

    def match_keywords_relationship(self, kw_list: List[str], k: int, threshold: float = 0.2, takes: int = 1):
        return self.__match_keywords(kw_list, k, self.vector_store_relationship, threshold=threshold, takes=takes)

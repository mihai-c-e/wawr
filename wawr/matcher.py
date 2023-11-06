import pandas as pd
from wawr.llm_interface import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import SKLearnVectorStore
import os

import logging

class KeywordMatchingService:
    def __init__(self, data_folder: str = None):
        self.embedder = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API'])
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

    def __match_keywords(self, kw_set, k, vector_store, threshold = 0.2):
        to_ret = {kw: [r for r in vector_store.similarity_search_with_score(kw, k=k) if r[1] < threshold] for kw in kw_set}
        return to_ret
    def match_keywords_node(self, kw_set, k):
        return self.__match_keywords(kw_set, k, self.vector_store_node)

    def match_keywords_relationship(self, kw_set, k):
        return self.__match_keywords(kw_set, k, self.vector_store_relationship)

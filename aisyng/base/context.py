from __future__ import annotations

from typing import List

from aisyng.base.datastore.base import PersistenceInterface
from aisyng.base.embeddings import EmbeddingPool
from aisyng.base.llms.base import LLMProviderPool
from aisyng.base.models import GraphElement
import logging


class AppContext:
    persistence: PersistenceInterface
    embedding_pool: EmbeddingPool
    llm_providers: LLMProviderPool
    def __init__(
            self,
            persistence: PersistenceInterface,
            embedding_pool: EmbeddingPool,
            llm_providers: LLMProviderPool
    ):
        self.persistence = persistence
        self.embedding_pool = embedding_pool
        self.llm_providers = llm_providers

    def add_graph_elements_embedding(
            self,
            embedding_key: str,
            element: GraphElement = None,
            elements: List[GraphElement] = None,
    ) -> GraphElement | List[GraphElement]:
        if (elements is None) == (element is None):
            raise ValueError("One and only one of element / elements arguments must be provided")
        if elements is None:
            elements = [element]
        logging.info(f"Calculating embeddings for {len(elements)} node(s)")
        embedder = self.embedding_pool.get_embedder(embedding_key)
        embeddings = embedder.create_embeddings([element.text])
        for i in range(len(embeddings)):
            elements[i].embeddings[embedder.name] = embeddings[i]
        return element if element is not None else elements
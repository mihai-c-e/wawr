from __future__ import annotations
from enum import Enum
from typing import cast, Callable, Any

from sqlalchemy import Engine

from aisyng.base.context import AppContext
from aisyng.wawr.wawr_embeddings import TextEmbedding3Small, TextEmbedding3Small128
from aisyng.base.embeddings import EmbeddingPool
from aisyng.wawr.persistence import WAWRPersistence
from aisyng.base.datastore.psql import PSQLPersistenceInterface
from aisyng.base.datastore.neo4j import Neo4JPersistenceInterface
from aisyng.base.llms.base import LLMProviderPool
from aisyng.base.llms.openai import OpenAIProvider
from aisyng.base.llms.mistral import MistralProvider
from aisyng.base.models.payload import TopicMeta
from aisyng.wawr.models.payload import DirectSimilarityTopicSolver
from aisyng.wawr.models.graph import PaperAbstract, Fact, Entity


class WAWRLLMProviders(str, Enum):
    OPENAI = "openai"
    MISTRAL = "mistral"

class WAWRContext(AppContext):

    @classmethod
    def create_default(cls, session_factory: Callable[[Any], Engine] = None) -> WAWRContext:
        openai_provider = OpenAIProvider()
        mistral_provider = MistralProvider()
        llm_providers = LLMProviderPool(
            providers = {
                WAWRLLMProviders.OPENAI: openai_provider,
                WAWRLLMProviders.MISTRAL: mistral_provider
            }
        )

        embedding_pool = EmbeddingPool()
        embedding_pool.add_embedder(TextEmbedding3Small(llm_provider=openai_provider))
        embedding_pool.add_embedder(TextEmbedding3Small128(llm_provider=openai_provider))

        payload_types = {TopicMeta, DirectSimilarityTopicSolver, PaperAbstract, Fact, Entity}
        persistence = WAWRPersistence(
            sqli = PSQLPersistenceInterface(
                embedding_pool=embedding_pool,
                payload_types=payload_types,
                session_factory=session_factory
            ),
            neo4ji = Neo4JPersistenceInterface(payload_types=payload_types)
        )

        return WAWRContext(
            llm_providers=llm_providers,
            persistence=persistence,
            embedding_pool=embedding_pool
        )

    def get_persistence(self) -> WAWRPersistence:
        return cast(WAWRPersistence, self.persistence)

    def get_embedding_pool(self) -> EmbeddingPool:
        return self.embedding_pool

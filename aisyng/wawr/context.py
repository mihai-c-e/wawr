from typing import cast

from aisyng.base.context import AppContext
from aisyng.wawr.wawr_embeddings import TextEmbedding3Small, TextEmbedding3Small128
from aisyng.base.embeddings import EmbeddingPool
from aisyng.wawr.persistence import WAWRPersistence
from aisyng.base.persistence.psql import PSQLPersistenceInterface
from aisyng.base.persistence.neo4j import Neo4JPersistenceInterface


class WAWRContext(AppContext):

    @classmethod
    def create_default(cls):
        embedding_pool = EmbeddingPool()
        embedding_pool.add_embedder(TextEmbedding3Small())
        embedding_pool.add_embedder(TextEmbedding3Small128())

        persistence = WAWRPersistence(
            sqli = PSQLPersistenceInterface(embedding_pool=embedding_pool),
            neo4ji = Neo4JPersistenceInterface()
        )

        return WAWRContext(
            persistence=persistence,
            embedding_pool=embedding_pool
        )

    def get_persistence(self) -> WAWRPersistence:
        return cast(WAWRPersistence, self.persistence)

    def get_embedding_pool(self) -> EmbeddingPool:
        return self.embedding_pool

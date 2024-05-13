from aisyng.base.persistence.base import PersistenceInterface
from aisyng.base.embeddings import EmbeddingPool

class AppContext:
    persistence: PersistenceInterface
    embedding_pool: EmbeddingPool

    def __init__(self, persistence: PersistenceInterface, embedding_pool: EmbeddingPool):
        self.persistence = persistence
        self.embedding_pool = embedding_pool
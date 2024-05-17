from typing import List

from aisyng.base.embeddings import Embedder
from aisyng.base.datastore.sqla import SQLABase
from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from aisyng.base.llms.base import LLMProvider

class OpenAITextEmbedding3Small(SQLABase):
    __tablename__ = "openai_text_embedding_3_small"
    node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"), primary_key=True)
    embedding: Mapped[List[float]] = mapped_column(Vector(1536))

class OpenAITextEmbedding3Small128(SQLABase):
    __tablename__ = "openai_text_embedding_3_small_128"
    node_id: Mapped[str] = mapped_column(ForeignKey("elements.id"), primary_key=True)
    embedding: Mapped[List[float]] = mapped_column(Vector(128))

class TextEmbedding3Small(Embedder):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            name="text-embedding-3-small",
            llm_provider=llm_provider,
            table=OpenAITextEmbedding3Small
        )

    def create_embeddings(self, data: List[str], **kwargs) -> List[List[float]]:
        return self.llm_provider.create_embeddings(data, model="text-embedding-3-small", **kwargs)

class TextEmbedding3Small128(Embedder):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            name="text-embedding-3-small-128",
            llm_provider=llm_provider,
            table=OpenAITextEmbedding3Small128
        )

    def create_embeddings(self, data: List[str], **kwargs) -> List[List[float]]:
        kwargs['dimensions'] = 128
        return self.llm_provider.create_embeddings(data, model="text-embedding-3-small", **kwargs)


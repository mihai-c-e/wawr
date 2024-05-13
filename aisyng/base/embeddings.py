from typing import Optional, List, Dict
from pydantic import BaseModel
from sqlalchemy.orm import DeclarativeBase

class Embedder:
    name: str
    table: Optional[DeclarativeBase]

    def __init__(self, name: str, table: DeclarativeBase = None):
        self.name = name
        self.table = table

    def create_embeddings(self, data: List[str], **kwargs) -> List[List[float]]:
        raise NotImplementedError()


class EmbeddingPool:
    embedders: Dict[str, Embedder] = dict()

    def add_embedder(self, embedder: Embedder) -> None:
        self.embedders[embedder.name] = embedder

    def get_embedder(self, embedding_key: str) -> Embedder:
        if not embedding_key in self.embedders:
            raise ValueError(f"Embedder with key {embedding_key} not found")
        return self.embedders[embedding_key]

    def embed(self, data: List[str], embedding_key) -> List[List[float]]:
        return self.get_embedder(embedding_key).create_embeddings(data)

    def get_table(self, embedding_key: str) -> DeclarativeBase:
        return self.get_embedder(embedding_key).table

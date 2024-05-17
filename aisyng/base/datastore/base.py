from datetime import datetime
from typing import List, Any

from aisyng.base.models import GraphElement, ScoredGraphElement
from aisyng.base.embeddings import Embedder


class PersistenceInterface:

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        raise NotImplementedError()

    def find_by_similarity(
            self,
            with_strings: List[str],
            with_vectors: List[List[float]],
            distance_threshold: float,
            embedder: Embedder,
            limit: int,
            from_date: datetime = None,
            to_date: datetime = None,
            only_type_ids: List[str] = None,
            exclude_type_ids: List[str] = None,
            **kwargs
    ) -> List[ScoredGraphElement]:
        raise NotImplementedError()

    def get_paths_between(
            self,
            from_node_ids: List[str],
            to_node_label: str,
            via_relationships: List[str],
            **kwargs
    ) -> Any:
        raise NotImplementedError()

class MultiMediaPersist(PersistenceInterface):

    media_list: List[PersistenceInterface]

    def __init__(self, media_list: List[PersistenceInterface]):
        self.media_list = media_list

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        return all([m.persist(objects_add=objects_add, objects_merge=objects_merge, **kwargs) for m in self.media_list])
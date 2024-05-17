from __future__ import annotations

from datetime import datetime
from typing import List, Any, Type, Dict, Set, TypeVar

from aisyng.base.models.graph import GraphElement, ScoredGraphElement, GraphElementTypes
from aisyng.base.embeddings import Embedder
from aisyng.base.models.base import PayloadBase


class PersistenceInterface:
    payload_types: Set[PayloadBase.__class__] = {}

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        raise NotImplementedError()

    def model_validate_payload(self, as_dict: Dict[str, Any]) -> PayloadBase | Dict[str, Any]:
        for payload_type in self.payload_types:
            attempted_object = payload_type.model_validate_or_none(as_dict)
            if attempted_object is not None:
                return attempted_object
        return as_dict

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
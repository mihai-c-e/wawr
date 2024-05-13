from typing import List

from aisyng.base.models import GraphElement


class PersistenceInterface:

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        raise NotImplementedError()

class MultiMediaPersist(PersistenceInterface):

    media_list: List[PersistenceInterface]

    def __init__(self, media_list: List[PersistenceInterface]):
        self.media_list = media_list

    def persist(self, objects_add: List[GraphElement] = None, objects_merge: List[GraphElement] = None,
                **kwargs) -> bool:
        return all([m.persist(objects_add=objects_add, objects_merge=objects_merge, **kwargs) for m in self.media_list])
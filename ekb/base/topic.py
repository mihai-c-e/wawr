from datetime import datetime
from ekb.base.models import GraphNode, GraphRelationship

class TopicNode(GraphNode):
    def __init__(self, text: str, **kwargs):
        meta = kwargs.pop("meta", dict())
        if not "progress" in meta:
            meta["progress"] = 0.0
        if not "status" in meta:
            meta["status"] = "initialized"
        if not "distance_threshold" in meta:
            meta["distance_threshold"] = 0.1
        super().__init__(text, **kwargs)

class TopicMatchRelationship(GraphRelationship):
    def __init__(self, **kwargs):
        super().__init__(text="matches", **kwargs)



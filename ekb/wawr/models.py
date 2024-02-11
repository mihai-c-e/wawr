import pandas as pd
from ekb.base.models import GraphNode, GraphRelationship


class PaperAbstract(GraphNode):
    @classmethod
    def from_series(self, source: pd.Series) -> "PaperAbstract":
        source_dict = source.to_dict()
        return PaperAbstract(
            id = source_dict.pop("id"),
            text=source_dict.pop("abstract"),
            date=source_dict.pop("date"),
            meta=source_dict
        )

    def __init__(self, text: str, **kwargs):
        super().__init__(text, **kwargs)


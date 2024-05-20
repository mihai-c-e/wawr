from __future__ import annotations

from datetime import datetime
from typing import List, Any, Iterable, cast, Dict, Set

from jinja2 import Template
from pydantic import BaseModel
from aisyng.base.models.graph import GraphNode, ScoredGraphElement, GraphElement, GraphElementTypes
from aisyng.base.models.payload import DirectSimilarityTopicSolverBase
from aisyng.base.utils import strftime_ymd
from aisyng.wawr.models.graph import WAWRGraphElementTypes


def node_to_reference_prompt_part(node: GraphNode, index: int) -> str:
    if node.type_id == WAWRGraphElementTypes.Abstract:
        return abstract_node_to_reference_prompt(node, index)
    if node.type_id == WAWRGraphElementTypes.Fact:
        return fact_node_to_reference_prompt(node, index)
    raise NotImplementedError()

def abstract_node_to_reference_prompt(node: GraphNode, index: int) -> str:
    return f'{index}. Paper date: {strftime_ymd(node.date)}\nPaper title: "{node.title}"\nPaper abstract: "{node.text}"'

def fact_node_to_reference_prompt(node: GraphNode, index: int) -> str:
    return (f'{index}. {node.text_type} in paper {node.title} from {strftime_ymd(node.date)}:\n'
            f'"{node.text}"')

def nodes_to_reference_prompt_part(nodes: List[GraphNode]) -> str:
    return "\n\n".join(
        [node_to_reference_prompt_part(node, index+1) for index, node in enumerate(nodes)]
    )

class TopicBreakdown(BaseModel):
    filter: List[str]
    questions: List[str]


class TopicReference(BaseModel):
    index: int
    text: str
    fact_type: str
    citation: str
    date: datetime
    title: str
    url: str
    similarity: float

    @classmethod
    def from_graph_node(cls, node: GraphNode, index: int = -1) -> "TopicReference":
        if node.type_id == WAWRGraphElementTypes.Abstract:
            return cls.from_abstract(node)
        return TopicReference(
            index=index,
            text=node.text,
        )

    @classmethod
    def from_abstract(cls, node: GraphNode, index: int = -1) -> "TopicReference":
        return TopicReference(
            index=index,
            test=node.text,
            fact_type="abstract",
            citation=None,
            date=node.date,

        )

class DirectSimilarityTopicSolver(DirectSimilarityTopicSolverBase):
    type_id: str = GraphElementTypes.TopicSolver

    def __init__(self, **data: Any):
        data["prompt_template"] = ("This is a set of reference extracts from research papers, ordered by date:\n"
                                   "{{ references_as_text }}\n\n"
                                   "Based on the knowledge from the references above, infer an answer as complete as "
                                   "possible to the following ask: "
                                   "\"{{ question }}\".\n"
                                   f"Address what is explicitly stated in the given references, then try to infer "
                                   f"a conclusion. Use the references and nothing else. If the referenecs"
                                   f"are missing or if they are irrelevant, answer with this. "
                                   f"Do not output anything after the conclusion. Format the text "
                                   f"using html tags, ready to insert as-is into a html page, using text formatting. "
                                   f"Quote the facts above in your answer in [1][2] format. "
                                   f"Do not list references, only use numbers in your answer to refer to the facts. "
                                   f"Write in concise style, in British English, but be very thorough take into account "
                                   f"all relevant references. If the references are not relevant for the ask, say so. "
                                   f"Do not write references or bibliography at the end. "
                                   f"Do not write references, only insert indexes towards given references."
                                   )
        data["solver_type"] = "direct_similarity"
        super().__init__(**data)

    def nodes_to_references_prompt_part(self, nodes: List[GraphNode]):
        return nodes_to_reference_prompt_part(nodes)

    def refine_search(self, matched_nodes: List[ScoredGraphElement]) -> List[List[GraphElement]]:
        return self._context.persistence.get_paths_between(
            from_node_ids=[mn.element.id for mn in matched_nodes if mn.element.type_id == WAWRGraphElementTypes.Fact],
            to_node_labels=[
                WAWRGraphElementTypes.Abstract,
                WAWRGraphElementTypes.FactType,
                # WAWRGraphElementTypes.Entity
            ],
            via_relationships=[
                WAWRGraphElementTypes.IsExtractedFrom,
                WAWRGraphElementTypes.IsTitleOf,
                WAWRGraphElementTypes.IsA
            ],
            max_hops=2
        )

    def select_references(self, graph_elements: Iterable[GraphElement]) -> List[GraphNode]:
        facts = set()
        abstracts = set()
        abstracts_dict: Dict[str, Dict[str, Any]] = {
            element.id: {'abstract':element, 'facts': list()}
            for element in graph_elements if element.type_id == WAWRGraphElementTypes.Abstract
        }
        force_abstract_ids: Set[str] = set()
        for element in graph_elements:
            if element.type_id == WAWRGraphElementTypes.Fact:
                abstracts_dict[element.source_id]['facts'].append(element)
            elif element.type_id == WAWRGraphElementTypes.Title:
                # If we matched a title, the abstract goes into references automatically
                force_abstract_ids.add(element.source_id)
        for abstract_id, d in abstracts_dict.items():
            if len(d['facts']) != 1 or abstract_id in force_abstract_ids:
                abstracts.add(d['abstract'])
            else:
                facts.add(d['facts'][0])
        return list(facts) + list(abstracts)
        """return cast(
            List[GraphNode]
,            [element for element in graph_elements if element.type_id == WAWRGraphElementTypes.Abstract]
        )"""

    def query_model(self, question: str, references: str, **kwargs) -> str:
        prompt = Template(self.prompt_template).render(
            question=question,
            references_as_text=references
        )
        llm_provider = self._context.llm_providers.get_by_model_name(self.llm_name)
        response, usage = llm_provider.query_model(query=prompt, model=self.llm_name)
        return response


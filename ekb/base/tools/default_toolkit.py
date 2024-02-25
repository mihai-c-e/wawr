import logging
from threading import Thread
from typing import List, Tuple
from datetime import datetime
from pydantic import BaseModel
from jinja2 import Template
from ekb.base.tools.sql_interface import SQLAElement, sql_to_element_list, persist_graph_elements
from ekb.base.models import GraphElement, GraphNode, GraphRelationship
from ekb.base.topic import TopicSolverBase, TopicNode, TopicMatchRelationship, TopicMeta
from ekb.base.tools.openai_models import create_embeddings, query_model, moderate_text, InappropriateContentException


class TopicReference(BaseModel):
    index: int
    text: str
    fact_type: str
    citation: str
    date: datetime
    title: str
    url: str
    similarity: float


def create_topic_node(topic: str, meta: TopicMeta) -> TopicNode:
    logging.info(f"Creating topic node: {topic} with meta: {meta}")
    topic_node = TopicNode(text=topic, topic_meta=meta)
    return topic_node

def add_element_embedding(element: GraphElement, embedding_key: str) -> GraphElement:
    logging.info(f"Calculating embeddings for node {element.id}: {element.text[:100]}")
    if embedding_key in element.embeddings:
        return element
    data = [element.text]
    embedding = create_embeddings(data, model=embedding_key)[0]
    element.embeddings[embedding_key] = embedding
    return element

def create_similarity_relationships(node: TopicNode, embedding_key: str) -> List[TopicMatchRelationship]:
    logging.info(f"Finding similar elements for node {node.id}: {node.text[:100]}")
    topic_meta = TopicMeta.model_validate(node.meta)
    similar_elements_sql = SQLAElement.find_by_similarity(
        with_string=node.text,
        with_vector=node.embeddings[embedding_key],
        distance_threshold=topic_meta.distance_threshold,
        embedding_key=topic_meta.embedding_key,
        limit=topic_meta.limit * 3,
        from_date=topic_meta.get_from_date(),
        to_date=topic_meta.get_to_date()
    )
    similar_elements = sql_to_element_list([e[0] for e in similar_elements_sql])
    scores = [e[1] for e in similar_elements_sql]
    logging.info(f"Found {len(similar_elements)} similar elements for node {node.id}: {node.text[:100]}")
    relationships = list()
    for i, element in enumerate(similar_elements):
        if isinstance(element, GraphNode):
            rel = TopicMatchRelationship(
                from_node=node,
                to_node=element,
                score=scores[i],
                match_type="cosine_similarity"
            )
            relationships.append(rel)
    return relationships

def identify_reference_nodes(node: TopicNode, subgraph: List[GraphElement]) -> Tuple[List[GraphElement], List[float]]:
    logging.info(f"Selecting references from subgraph for node {node.id}: {node.text[:100]}")
    references_list = list()
    for element in subgraph:
        if isinstance(element, TopicMatchRelationship) and (element.to_node.type_id in ["Fact", "PaperAbstract"]):
            references_list.append((element.to_node, element.meta['score']))
    references_list.sort(key=lambda x: x[1])
    logging.info(f"Selected {len(references_list)} from subgraph for node {node.id}: {node.text[:100]}")
    reference_nodes = [x[0] for x in references_list]
    reference_scores = [x[1] for x in references_list]
    return reference_nodes, reference_scores

def limit_references(
        meta: TopicMeta,
        reference_nodes: List[GraphElement],
        reference_scores: List[float],
) -> Tuple[List[GraphElement], List[float]]:
    limit = meta.limit
    if len(reference_nodes) > limit:
        logging.info(f"Reducing references from {len(reference_nodes)} to {limit}")
        refs = list(zip(reference_nodes, reference_scores))
        refs.sort(key=lambda x: x[0].date)
        refs = refs[::-1]
        new_refs = refs[:limit]
        date_to = new_refs[0][0].date
        date_from = new_refs[-1][0].date
        logging.info(f"Reduced to {limit} references between "
                     f"{date_from.strftime('%Y-%m-%d')} and {date_to.strftime('%Y-%m-%d')}")
        meta.user_message += (f"Your search returned too many elements from the knowledge graph. "
                              f"I reduced them to get 200 references by "
                              f"setting the retrieval period between {date_from.strftime('%d %b %Y')} "
                              f"and {date_to.strftime('%d %b %Y')}. Try to increase precision "
                              f"to get less records or repeat the question "
                              f"on a different interval."
        )
        reference_nodes = [x[0] for x in refs[:limit]]
        reference_scores = [x[1] for x in refs]
    elif len(reference_nodes) > 50:
        meta.user_message = ("We identified a large nubmer of references. Please note that "
                             "some of them might be ignored by the model when generating an answer.")
    return reference_nodes, reference_scores

def create_references(node: TopicNode, reference_nodes: List[GraphElement], reference_scores: List[float]) -> List[TopicReference]:
    references = list()
    for i, n in enumerate(zip(reference_nodes, reference_scores)):
        similarity = n[1]
        ref_node = n[0]
        ref = TopicReference(
            index=i+1,
            text=ref_node.text,
            citation=ref_node.meta.get("citation", ""),
            title=ref_node.meta["title"],
            date=ref_node.date,
            url="",
            similarity=similarity,
            fact_type=""
        )
        references.append(ref)
    return references

def references_to_prompt_text(references: List[TopicReference]) -> str:
    return "\n".join([
        f"{r.index}. In the paper '{r.title}' from '{r.date}': {r.text}"
        for r in references
    ])

def get_answer(node: TopicNode, references: List[TopicReference]):
    prompt_template = """
    This is what we know:
    {{ references_as_text }}
    
    Based on the references above, infer an answer to the following question: 
    "{{ question }}".
    
    Start with what is explicitly stated in references regarding the question, then try to infer a summary.
    Format your answer using html tags, ready to insert as-is into a html page. Provide index references in text in 
    [1][2] format. Be thorough in your response, trying to take into account every relevant reference - but,
    if there are too many references, warn the user that the answer might be incomplete.
     
    """
    prompt = Template(prompt_template).render(
        references_as_text = references_to_prompt_text(references),
        question = node.text
    )
    response = query_model(query=prompt, model=node.get_topic_meta().model)
    return response

def _topic_solver_v1(node: TopicNode) -> TopicNode:
    try:
        meta = node.get_topic_meta()

        moderate_text(node.text)

        add_element_embedding(element=node, embedding_key=meta.embedding_key)
        meta.status = "Retrieving"
        meta.progress = 0.1
        meta.log_history.append("Calculated embeddings")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])

        relationships = create_similarity_relationships(node=node, embedding_key=meta.embedding_key)
        meta.subgraph_ids = [rel.id for rel in relationships]
        meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
        meta.status = "References"
        meta.progress = 0.5
        meta.log_history.append("Retrieved subgraph")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node], elements_add=relationships)

        reference_nodes, reference_scores = identify_reference_nodes(node=node, subgraph=relationships)
        reference_nodes, reference_scores = limit_references(meta=meta, reference_nodes=reference_nodes, reference_scores=reference_scores)
        meta.reference_ids = [n.id for n in reference_nodes]
        meta.reference_scores = [s for s in reference_scores]
        references = create_references(node=node, reference_nodes=reference_nodes, reference_scores=reference_scores)

        meta.status = "Answering"
        meta.progress = 0.7
        meta.log_history.append("Retrieved references")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])

        if len(references) == 0:
            meta.response = "No data found. Try relaxing the parameters (e.g. larger time interval, or lower precision."
            meta.usage = {}
        else:
            response = get_answer(node=node, references=references)
            meta.response = response[0]
            meta.usage = dict(response[1].usage)
        meta.status = "Completed"
        meta.progress = 1.0
        meta.log_history.append("Answer received")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])
        return node
    except InappropriateContentException as ex:
        meta.status = "Error"
        meta.progress = 0.0
        meta.log_history.append(f"Error: {str(ex)}")
        meta.user_message = "Your question was flagged as inappropriate and will not be processed."
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])
    except Exception as ex:
        meta.status = "Error"
        meta.progress = 0.0
        meta.log_history.append(f"Error: {str(ex)}")
        node.update_topic_meta(meta)
        meta.user_message = ("There was an error processing your request. Please try again."
                             "If the error persist, we would appreciate your reporting this to "
                             "contact@wawr.ai (include the link). Thank you.")
        persist_graph_elements(elements_merge=[node])
        raise

def topic_solver_v1(topic: str = None, meta: TopicMeta = None, node: TopicNode = None, in_thread: bool = False) -> TopicNode:
    if (topic is None or meta is None) and node is None:
        raise ValueError("Either topic and meta, or node must be provided")
    try:
        if node is None:
            node = create_topic_node(topic=topic, meta=meta)
            meta.log_history.append("Initialised")
            meta.status = "Embedding"
            node.update_topic_meta(meta)
            persist_graph_elements(elements_add=[node])

        if not in_thread:
            return _topic_solver_v1(node)
        else:
            thread = Thread(
                name=f"Topic solver for node {node.id}: {node.text}",
                target=_topic_solver_v1,
                kwargs={"node": node},
            )
            thread.start()
            return node
    except Exception as ex:
        meta.status = "Error"
        meta.progress = 0.0
        meta.log_history.append(f"Error: {str(ex)}")
        raise

def read_html(s: str) -> str:
    if '```html' in s:
        s = s.split('```html')[1]
    if 'html```' in s:
        s = s.split('html```')[1]
    if '```' in s:
        s = s.split('```')[0]
    return s
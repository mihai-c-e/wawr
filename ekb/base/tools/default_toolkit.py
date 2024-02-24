import logging
from threading import Thread
from typing import List
from datetime import datetime
from pydantic import BaseModel
from jinja2 import Template
from ekb.base.tools.sql_interface import SQLAElement, sql_to_element_list, persist_graph_elements
from ekb.base.models import GraphElement, GraphNode, GraphRelationship
from ekb.base.topic import TopicSolverBase, TopicNode, TopicMatchRelationship, TopicMeta
from ekb.base.tools.openai_models import create_embeddings, query_model


class TopicReference(BaseModel):
    index: int
    text: str
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
        limit=topic_meta.limit
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

def identify_reference_nodes(node: TopicNode, subgraph: List[GraphElement]) -> List[GraphElement ]:
    logging.info(f"Selecting references from subgraph for node {node.id}: {node.text[:100]}")
    references_list = list()
    for element in subgraph:
        if isinstance(element, TopicMatchRelationship) and element.to_node.type_id=="Fact":
            references_list.append(element)
    references_list.sort(key=lambda x: x.meta["score"])
    logging.info(f"Selected {len(references_list)} from subgraph for node {node.id}: {node.text[:100]}")
    return references_list

def create_references(node: TopicNode, reference_nodes: List[GraphElement]) -> List[TopicReference]:
    references = list()
    for i, n in enumerate(reference_nodes):
        if isinstance(n, TopicMatchRelationship):
            similarity = n.meta["score"]
            n = n.to_node
        else:
            similarity = 1
        ref = TopicReference(
            index=i+1,
            text=n.text,
            citation=n.meta.get("citation", ""),
            title=n.meta["title"],
            date=n.date,
            url="",
            similarity=similarity
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
        add_element_embedding(element=node, embedding_key=meta.embedding_key)
        meta.status = "Retrieving similar elements"
        meta.progress = 0.1
        meta.log_history.append("Calculated embeddings")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])

        relationships = create_similarity_relationships(node=node, embedding_key=meta.embedding_key)
        meta.subgraph_ids = [rel.id for rel in relationships]
        meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
        meta.status = "Building references"
        meta.progress = 0.5
        meta.log_history.append("Retrieved subgraph")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node], elements_add=relationships)

        reference_nodes = identify_reference_nodes(node=node, subgraph=relationships)
        references = create_references(node=node, reference_nodes=reference_nodes)
        meta.subgraph_ids = [rel.id for rel in relationships]
        meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
        meta.status = "Getting answer"
        meta.progress = 0.7
        meta.log_history.append("Retrieved references")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])

        response = get_answer(node=node, references=references)
        meta.response = response[0]
        meta.usage = dict(response[1].usage)
        meta.status = "Completed"
        meta.progress = 1.0
        meta.log_history.append("Answer received")
        node.update_topic_meta(meta)
        persist_graph_elements(elements_merge=[node])
        return node
    except Exception as ex:
        meta.status = "Error"
        meta.progress = 0.0
        meta.log_history.append(f"Error: {str(ex)}")
        raise

def topic_solver_v1(topic: str = None, meta: TopicMeta = None, node: TopicNode = None, in_thread: bool = False) -> TopicNode:
    if (topic is None or meta is None) and node is None:
        raise ValueError("Either topic and meta, or node must be provided")
    try:
        if node is None:
            node = create_topic_node(topic=topic, meta=meta)
            meta.log_history.append("Initialised")
            meta.status = "Calculating embeddings"
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

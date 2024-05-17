import json
import logging
import time

import numpy as np
from threading import Thread
from typing import List, Tuple, Dict, Any
from jinja2 import Template
from aisyng.base.tools.sql_interface import SQLAElement, SQLToolkit
from aisyng.wawr.models.payload import TopicBreakdown, TopicReference
from base.models.payload import TopicMeta, TopicMatchRelationship
from aisyng.base.embeddings import EmbeddingPool


class EKBToolkit:
    embedding_pool: EmbeddingPool
    sql_toolkit: SQLToolkit

    def __init__(self, embedding_pool: EmbeddingPool):
        self.embedding_pool = embedding_pool
        self.sql_toolkit = SQLToolkit(embedding_pool=embedding_pool)

    def create_similarity_relationships(self, node: TopicNode, embedding_key: str) -> List[TopicMatchRelationship]:
        logging.info(f"Finding similar elements for node {node.id}: {node.text[:100]}")
        topic_meta = TopicMeta.model_validate(node.meta)
        embedder = self.embedding_pool.get_embedder(embedding_key)
        start = time.time()
        similar_elements_sql = SQLAElement.find_by_similarity(
            with_strings=[node.text],
            with_vectors=[node.embeddings[embedder.name]],
            distance_threshold=topic_meta.distance_threshold,
            embedder=embedder,
            limit=topic_meta.limit * 3,
            from_date=topic_meta.get_from_date(),
            to_date=topic_meta.get_to_date(),
            only_type_ids=["abstract", "fact"]
        )
        logging.info(f"DB time: {time.time() - start}")
        similar_elements = self.sql_toolkit.sql_to_element_list([e[0] for e in similar_elements_sql])
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

    def create_similarity_relationships_with_entities(self, node: TopicNode, entities: List[str], embedding_key: str) -> \
    List[TopicMatchRelationship]:
        logging.info(f"Finding similar elements for node {node.id}: {node.text[:100]}")
        topic_meta = node.get_topic_meta()
        embedder = self.embedding_pool.get_embedder(embedding_key)
        embeddings = embedder.create_embeddings(data=entities)
        similar_elements_sql = SQLAElement.find_by_similarity(
            with_strings=entities,
            with_vectors=embeddings,
            distance_threshold=topic_meta.distance_threshold,
            embedder=embedder,
            limit=topic_meta.limit * 3,
            from_date=topic_meta.get_from_date(),
            to_date=topic_meta.get_to_date(),
            only_type_ids=["fact", "abstract", "entity"]
        )
        similar_elements = self.sql_toolkit.sql_to_element_list([e[0] for e in similar_elements_sql])
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

    def create_similarity_relationships_with_hypothetical(self, node: TopicNode, hypothetical: List[str],
                                                          embedding_key: str) -> List[TopicMatchRelationship]:
        logging.info(f"Finding similar elements for node {node.id}: {node.text[:100]}")
        topic_meta = node.get_topic_meta()
        embedder = self.embedding_pool.get_embedder(embedding_key)
        embeddings = embedder.create_embeddings(data=hypothetical)
        embedding = np.mean(embeddings, axis=0)
        similar_elements_sql = SQLAElement.find_by_similarity(
            with_strings=hypothetical[:1],
            with_vectors=[embedding],
            distance_threshold=topic_meta.distance_threshold,
            embedder=embedder,
            limit=topic_meta.limit * 3,
            from_date=topic_meta.get_from_date(),
            to_date=topic_meta.get_to_date(),
            only_type_ids=["fact", "abstract", "entity"]
        )
        similar_elements = self.sql_toolkit.sql_to_element_list([e[0] for e in similar_elements_sql])
        scores = [e[1] for e in similar_elements_sql]
        logging.info(f"Found {len(similar_elements)} similar elements for node {node.id}: {node.text[:100]}")
        relationships = list()
        for i, element in enumerate(similar_elements):
            if isinstance(element, GraphNode):
                rel = TopicMatchRelationship(
                    from_node=node,
                    to_node=element,
                    score=scores[i],
                    match_type="cosine_similarity",

                )
                relationships.append(rel)
        return relationships


    def identify_reference_nodes(self, node: TopicNode, subgraph: List[GraphElement]) -> Tuple[List[GraphElement], List[float]]:
        logging.info(f"Selecting references from subgraph for node {node.id}: {node.text[:100]}")
        references_list = list()
        for element in subgraph:
            if isinstance(element, TopicMatchRelationship) and (element.to_node.type_id in ["fact", "abstract"]):
                references_list.append((element.to_node, element.meta['score']))
        references_list.sort(key=lambda x: x[1])
        logging.info(f"Selected {len(references_list)} from subgraph for node {node.id}: {node.text[:100]}")
        reference_nodes = [x[0] for x in references_list]
        reference_scores = [x[1] for x in references_list]
        return reference_nodes, reference_scores


    def limit_references(
            self,
            meta: TopicMeta,
            reference_nodes: List[GraphElement],
            reference_scores: List[float],
    ) -> Tuple[List[GraphElement], List[float]]:
        limit = meta.limit
        if len(reference_nodes) > limit:
            logging.info(f"Reducing references from {len(reference_nodes)} to {limit}")
            reference_nodes = reference_nodes[:limit]
            reference_scores = reference_scores[:limit]
            logging.info(f"Reduced to {limit} references ")
            meta.user_message += (
                f"There are too many data points in the knowledge graph matching your question with the selected parameters. "
                f"I limited my answer to take into account only the top {limit} references. Try to increase precision "
                f"to get less records or repeat the question "
                f"on a different interval."
            )
            """refs = list(zip(reference_nodes, reference_scores))
            refs.sort(key=lambda x: x[0].date)
            refs = refs[::-1]
            new_refs = refs[:limit]
            date_to = new_refs[0][0].date
            date_from = new_refs[-1][0].date
            new_refs.sort(key=lambda x: x[1])
            logging.info(f"Reduced to {limit} references between "
                         f"{date_from.strftime('%Y-%m-%d')} and {date_to.strftime('%Y-%m-%d')}")
            meta.user_message += (f"There are too many data points in the knowledge graph matching your question with the selected parameters. "
                                  f"I limited my answer to take into account 200 references "
                                  f"between {date_from.strftime('%d %b %Y')} "
                                  f"and {date_to.strftime('%d %b %Y')}. Try to increase precision "
                                  f"to get less records or repeat the question "
                                  f"on a different interval."
            )
            reference_nodes = [x[0] for x in new_refs]
            reference_scores = [x[1] for x in new_refs]"""

        elif len(reference_nodes) > 50:
            meta.user_message = ("We identified a large number of references. Sometimes, in such cases, the answer might "
                                 "omit relevant information.")
        return reference_nodes, reference_scores


    def create_references(self, node: TopicNode, reference_nodes: List[GraphElement], reference_scores: List[float]) -> List[
        TopicReference]:
        references = list()
        for i, n in enumerate(zip(reference_nodes, reference_scores)):
            similarity = n[1]
            ref_node = n[0]
            ref = TopicReference(
                index=i + 1,
                text=ref_node.text,
                citation=ref_node.citation,
                title=ref_node.title,
                date=ref_node.date,
                url="",
                similarity=similarity,
                fact_type=""
            )
            references.append(ref)
        return references


    def references_to_prompt_text(self, references: List[TopicReference]) -> str:
        return "\n".join([
            f"{r.index}. In the paper '{r.title}' from '{r.date}': {r.text}"
            for r in references
        ])


    def get_answer(self, node: TopicNode, references: List[TopicReference]):
        prompt_template = """
    This is a set of reference extracts from research papers, ordered by relevance:
    {{ references_as_text }}
        
    Based on the knowledge from the references above, infer an answer as complete as possible to the following ask: "{{ question }}".
        
    Address what is explicitly stated in the given references, then try to infer a conclusion. Do not output anything after the conclusion. Format the text using html tags, ready to insert as-is into a html page, using text formatting. Quote the facts above in your answer in [1][2] format. Do not list references, only use numbers in your answer to refer to the facts. Write in concise style, in British English, but be very thorough take into account all relevant references. If the references are not relevant for the ask, say so. Do not write references or bibliography at the end. Do not write references, only insert indexes towards given references.
    
    """
        prompt = Template(prompt_template).render(
            references_as_text=self.references_to_prompt_text(references),
            question=node.text
        )
        response = query_model(query=prompt, model=node.get_topic_meta().model)
        return response


    def break_down_question(self, question: str, model: str) -> TopicBreakdown:
        prompt_template = """
        Answer with json and only json as per below:
        For question: "Which language model performs best at code writing?", your output should be:
        {
            "filter": ["code generation"],
            "questions":[ 
                "What is the research on code generation with language models?",
                "Output a table of benchmark results for code generation with language models",
                "Which language model performs best at code writing?"
            ]
        }
        
        For question: "How to increase retrieval precision in retrieval-augmented generation?", your output should be:
        {
            "filter": ["retrieval augmented generation"],
            "questions": [
                "What is the research on retrieval augmented generation?",
                "How to increase retrieval precision in retrieval-augmented generation? "
            ]
        }    
        
        For question: "Which model is better between GPT 3.5 and Mixtral?", your output should be:
        {
            "filter": ["comparison", "GPT 3.5", "Mixtral"],
            "questions": [
                "Are there benchmarks that include both GPT 3.5 and Mixtral?",
                "Which model is better between GPT 3.5 and Mixtral?"
            ]
        }    
        
        For question: "Can language models play chess?", your output should be:
        {
            "filter": ["chess"],
            "questions": [
                "Can language models play chess?"
            ]
        }    
        For question: "What games can language models play?", your output should be:
        {
            "filter": ["model play game"],
            "questions": [
                "What games can language models play?"
            ]
        }    
        
        For question: "{{ question }}", your output is:
        
        """
        prompt = Template(prompt_template).render(question=question)
        response = query_model(query=prompt, model=model)
        result = TopicBreakdown.model_validate(read_json(response[0]))
        return result


    def find_hypothetical_answers(self, question: str, model: str) -> List[str]:
        prompt_template = f"""
            We need to answer the following ask: "{question}".
            The answer must be based on facts retrieved from a database. The facts are short sentences extracted from 
            research paper abstracts on language models. Provide at least 40 hypothetical facts that would
            answer the question, one per row, no bullets or numbers. If the ask does not mention specific entities,
            do not use any entities in your answer. If entities are mentioned, use them in the hypothetical facts.
            If the question includes an 'or', make sure to output facts for all variants. 
            Provide your answer:
            
            """
        prompt = Template(prompt_template).render(question=question)
        response = query_model(query=prompt, model=model)
        result = response[0].split('\n')
        return result


    def _topic_solver_v1(self, node: TopicNode, embedding_key: str) -> TopicNode:
        try:
            meta = node.get_topic_meta()
            moderate_text(node.text)
            self.add_element_embedding(element=node, embedding_key=embedding_key)
            meta.status = "Retrieving"
            meta.progress = 0.1
            meta.log_history.append("Calculated embeddings")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])

            relationships = self.create_similarity_relationships(node=node, embedding_key=embedding_key)
            meta.subgraph_ids = [rel.id for rel in relationships]
            meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
            meta.status = "References"
            meta.progress = 0.5
            meta.log_history.append("Retrieved subgraph")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node], elements_add=relationships)

            reference_nodes, reference_scores = self.identify_reference_nodes(node=node, subgraph=relationships)
            reference_nodes, reference_scores = self.limit_references(meta=meta, reference_nodes=reference_nodes,
                                                                 reference_scores=reference_scores)
            meta.reference_ids = [n.id for n in reference_nodes]
            meta.reference_scores = [s for s in reference_scores]
            references = self.create_references(node=node, reference_nodes=reference_nodes, reference_scores=reference_scores)

            meta.status = "Answering"
            meta.progress = 0.7
            meta.log_history.append("Retrieved references")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])

            if len(references) == 0:
                meta.response = "No data found. Try relaxing the parameters (e.g. larger time interval, or lower precision)."
                meta.usage = {}
            else:
                response = self.get_answer(node=node, references=references)
                meta.response = response[0]
                meta.usage = dict(response[1].usage)
            meta.status = "Completed"
            meta.progress = 1.0
            meta.log_history.append("Answer received")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
            return node
        except InappropriateContentException as ex:
            meta.status = "Error"
            meta.progress = 0.0
            meta.log_history.append(f"Error: {str(ex)}")
            meta.user_message = "Your question was flagged as inappropriate and will not be processed."
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
        except Exception as ex:
            meta.status = "Error"
            meta.progress = 0.0
            meta.log_history.append(f"Error: {str(ex)}")
            node.update_topic_meta(meta)
            meta.user_message = ("There was an error processing your request. Please try again."
                                 "If the error persist, we would appreciate your reporting this to "
                                 "contact@wawr.ai (include the link). Thank you.")
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
            raise


    def topic_solver_v1(self, embedding_key: str, topic: str = None, meta: TopicMeta = None, node: TopicNode = None,
                        in_thread: bool = False) -> TopicNode:
        if (topic is None or meta is None) and node is None:
            raise ValueError("Either topic and meta, or node must be provided")
        try:
            if node is None:
                node = self.create_topic_node(topic=topic, meta=meta)
                meta.log_history.append("Initialised")
                meta.status = "Embedding"
                node.update_topic_meta(meta)
                self.sql_toolkit.persist_graph_elements(elements_add=[node])

            if not in_thread:
                return self._topic_solver_v1(node=node, embedding_key=embedding_key)
            else:
                thread = Thread(
                    name=f"Topic solver for node {node.id}: {node.text}",
                    target=self._topic_solver_v1,
                    kwargs={"node": node},
                )
                thread.start()
                return node
        except Exception as ex:
            meta.status = "Error"
            meta.progress = 0.0
            meta.log_history.append(f"Error: {str(ex)}")
            raise


    def _topic_solver_v2(self, node: TopicNode, embedding_key: str) -> TopicNode:
        try:
            meta = node.get_topic_meta()
            moderate_text(node.text)
            meta.hypothetical = self.find_hypothetical_answers(node.text, model=meta.model)
            # meta.breakdown = dict(break_down_question(node.text, model=meta.model))
            self.add_element_embedding(element=node, embedding_key=embedding_key)
            meta.status = "Retrieving"
            meta.progress = 0.1
            meta.log_history.append("Calculated embeddings")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])

            relationships = self.create_similarity_relationships_with_hypothetical(
                node=node,
                hypothetical=meta.hypothetical,
                embedding_key=embedding_key,
            )
            meta.subgraph_ids = [rel.id for rel in relationships]
            meta.subgraph_ids.extend(rel.to_node.id for rel in relationships)
            meta.status = "References"
            meta.progress = 0.5
            meta.log_history.append("Retrieved subgraph")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node], elements_add=relationships)

            reference_nodes, reference_scores = self.identify_reference_nodes(node=node, subgraph=relationships)

            reference_nodes, reference_scores = self.limit_references(meta=meta, reference_nodes=reference_nodes,
                                                                 reference_scores=reference_scores)
            meta.reference_ids = [n.id for n in reference_nodes]
            meta.reference_scores = [s for s in reference_scores]
            references = self.create_references(node=node, reference_nodes=reference_nodes, reference_scores=reference_scores)

            meta.status = "Answering"
            meta.progress = 0.7
            meta.log_history.append("Retrieved references")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])

            if len(references) == 0:
                meta.response = "No data found. Try relaxing the parameters (e.g. larger time interval, or lower precision)."
                meta.usage = {}
            else:
                response = self.get_answer(node=node, references=references)
                meta.response = response[0]
                meta.usage = dict(response[1].usage)
            meta.status = "Completed"
            meta.progress = 1.0
            meta.log_history.append("Answer received")
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
            return node
        except InappropriateContentException as ex:
            meta.status = "Error"
            meta.progress = 0.0
            meta.log_history.append(f"Error: {str(ex)}")
            meta.user_message = "Your question was flagged as inappropriate and will not be processed."
            node.update_topic_meta(meta)
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
        except Exception as ex:
            meta.status = "Error"
            meta.progress = 0.0
            meta.log_history.append(f"Error: {str(ex)}")
            node.update_topic_meta(meta)
            meta.user_message = ("There was an error processing your request. Please try again."
                                 "If the error persist, we would appreciate your reporting this to "
                                 "contact@wawr.ai (include the link). Thank you.")
            self.sql_toolkit.persist_graph_elements(elements_merge=[node])
            raise


    def topic_solver_v2(self, embedding_key: str, topic: str = None, meta: TopicMeta = None, node: TopicNode = None,
                        in_thread: bool = False) -> TopicNode:
        if (topic is None or meta is None) and node is None:
            raise ValueError("Either topic and meta, or node must be provided")
        try:
            if node is None:
                node = self.create_topic_node(topic=topic, meta=meta)
                meta.log_history.append("Initialised")
                meta.status = "Embedding"
                node.update_topic_meta(meta)
                self.sql_toolkit.persist_graph_elements(elements_add=[node])

            if not in_thread:
                return self._topic_solver_v1(node=node, embedding_key=embedding_key)
            else:
                thread = Thread(
                    name=f"Topic solver for node {node.id}: {node.text}",
                    target=self._topic_solver_v2,
                    kwargs={"node": node, "embedding_key": embedding_key},
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


def read_json(s: str) -> Dict[Any, Any]:
    if '```json' in s:
        s = s.split('```json')[1]
    if 'json```' in s:
        s = s.split('json```')[1]
    if '```' in s:
        s = s.split('```')[0]
    # s = s.replace("\\", r"\\\\")
    return json.loads(s)

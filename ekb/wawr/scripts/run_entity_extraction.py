import re
from functools import partial
from typing import List, Dict, Tuple, Set
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy.orm import aliased

from ekb.base.models import GraphNode, GraphRelationship

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from jinja2 import Template
from pydantic import BaseModel
from ekb.base.tools.openai_models import query_model
import logging
from multiprocessing.pool import ThreadPool
import threading
from ekb.base.tools.sql_interface import element_to_sql, Session, SQLAElement, sql_to_element
from ekb.utils import read_json

sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}

class _Entity(BaseModel):
    name: str
    type: str

def _get_qualifying_facts(limit: int = 10) -> List[GraphNode]:
    abstract = aliased(SQLAElement)
    fact = aliased(SQLAElement)
    with (Session() as sess):
        stmt = select(fact, abstract.title, abstract.text).where(
            (fact.source_id == abstract.id)
            & (fact.type_id == "fact")
            & (~fact.status.contains('entities'))
        ).order_by(fact.date.desc()).limit(limit)
        result = sess.execute(stmt).all()
        data = [(sql_to_element(n[0]), n[1], n[2]) for n in result]
    if len(data) > 0:
        logging.info(f"Loaded {len(data)} facts")
    return data

def _save(master_node: GraphNode, nodes: Set[GraphNode], relationships: List[GraphRelationship]):
    logging.info(f"Saving {len(nodes)} nodes and {len(relationships)} relationships")
    with sql_lock:
        logging.info("Lock acquired")
        master_node_sql = element_to_sql(master_node)
        nodes_sql = [element_to_sql(n) for n in nodes]
        relationships_sql = [element_to_sql(r) for r in relationships]
        with Session() as sess:
            with sess.begin():
                sess.merge(master_node_sql)
                for node in nodes_sql:
                    sess.merge(node)
                for rel in relationships_sql:
                    sess.merge(rel)

    logging.info(f"Saving complete")

def _get_model_response(data: Tuple[GraphNode, str, str], model: str, max_attempts: int = 5) -> List[_Entity]:
    node, title, abstract = data
    logging.info(f"Extracting entities for fact: \"{node.text}\"/\"{title}\"")
    str_template = """
        Research paper title: {{ title }}
        Research paper abstract: "
        {{ abstract }}        
        "
        Text of interest from the abstract above: "{{ text }}"
        Output at least 2 and at most 5 entities from the abstract that are implicitly or explicitly mentioned in the text of interest. 
        Provide a relevant type for each entity, such as: model, language model, algorithm, dataset, benchmark or others.
        Reduce the entity names to a minimal form, for example: "large language model" instead of 
        "large language models (LLM)" or "LLM". Use these entity types when possible: "large language model",
        "language model", "algorithm", "benchmark", "dataset", "result", "model family".
        Make sure the entities are relevant for the given text of interest, not just for the entire abstract.
        Output in json format as the example below. Output json and only json:
        [
        {"name":"...", "type":"..."},
        {"name":"...", "type":"..."}
        ]        
        """
    template = Template(str_template)
    text = node.text
    query = template.render(text=text, title=title, abstract=abstract)
    attempt = 1
    while True:
        try:
            response, completion = query_model(query, model=model)
            with openai_api_lock:
                for k, v in dict(completion.usage).items():
                    openai_api_usage[k] = openai_api_usage.get(k, 0) + v
                logging.info(f"OpenAI API usage: {openai_api_usage}")

            response = read_json(response)
            extracted_objects = [_Entity.model_validate(r) for r in response]
            break
        except Exception as ex:
            logging.exception(ex)
            attempt += 1
            if attempt > max_attempts:
                raise
            logging.error(f"Attempt {attempt}")
    logging.info(f"Model responded with {len(extracted_objects)} facts")
    return extracted_objects

def _extracted_object_to_nodes_and_relationship(
        fact_node: GraphNode, entity_obj: _Entity, nodes: Set[GraphNode], relationships: List[GraphRelationship]
) -> None:
    # Add entity node
    entity_node = GraphNode(id=f"entity:{entity_obj.name}", text=entity_obj.name, type_id="entity", text_type=entity_obj.type)
    nodes.add(entity_node)
    # Add relationship between entity node and fact
    relationships.append(GraphRelationship(
        id=f"{entity_node.id}-{fact_node.id}", from_node=entity_node, to_node=fact_node, text="mentioned in", type_id="mentioned_in"
    ))
    # Add entity type node
    entity_type_node = GraphNode(
        id=f"entity_type:{entity_obj.type}", text=entity_obj.type, type_id="entity_type", text_type="entity_type"
    )
    nodes.add(entity_type_node)
    # Add the is_a relationship between entity type node and entity node
    relationships.append(GraphRelationship(
        id=f"{entity_node.id}-{entity_type_node.id}", from_node=entity_node, to_node=entity_type_node, text="is a", type_id="is_a"
    ))

def _extract_from_one(data: Tuple, model: str) -> Tuple[Set[GraphNode], List[GraphRelationship]]:
    global openai_api_lock
    global sql_lock
    global openai_api_usage
    fact_node = data[0]
    extracted_objects = _get_model_response(data, max_attempts=5, model=model)
    nodes = set()
    relationships = list()

    for entity_obj in extracted_objects:
        _extracted_object_to_nodes_and_relationship(fact_node, entity_obj, nodes, relationships)

    fact_node.status += "entities "
    logging.info(f"Created {len(nodes)} nodes and {len(relationships)} relationships")
    _save(fact_node, nodes, relationships)
    return nodes, relationships

def perform_extraction( model: str, max_count: int = 20, pool_size: int = 1):
    data = _get_qualifying_facts(max_count)
    extract_from_one_partial = partial(_extract_from_one, model=model)
    pool = ThreadPool(pool_size)
    pool.map(extract_from_one_partial, data)
    pool.close()
    pool.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # model="gpt-4-0125-preview"
    model = "gpt-3.5-turbo-0125"
    perform_extraction(model=model, max_count=10000, pool_size=100)
    logging.info("Entity extraction done, exiting...")


import json
import random
from functools import partial
from typing import List, Tuple, Set
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy.orm import aliased

from aisyng.base.models import GraphNode, GraphRelationship, GraphElement

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from jinja2 import Template
from pydantic import BaseModel
from base.llms.openai_models import query_model
import logging
from multiprocessing.pool import ThreadPool
import threading
from aisyng.utils import read_json

sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
entity_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}
sql_toolkit = SQLToolkit(embedding_pool=embedding_pool)
entity_list = list()


entity_extraction_template1 = (
        "Examples of entities:\n"
        "\"{{ entity_samples }}\"\n\n"        
        "Research paper title: \"{{ title }}\"\n\n"        
        "Research paper abstract: \n\""
        "{{ abstract }}"        
        "\"\n\n"
        "Facts extracted from the abstract above:\n"
        " {% for fact in facts %}{{ loop.index }}.{{fact.text}}\n{% endfor %}\n\n"
        "For each fact, identify and output a json of entities and their types. "
        "Be thorough and aim to provide between 2 and 5 entities for each fact. The type has to be relevant "
        "for each entity, such as: model, language model, algorithm, dataset, benchmark or others."
        "Reduce the entity names to a minimal form, for example: \"large language model\" instead of" 
        "\"large language models (LLM)\" or \"LLM\". Use these entity types when possible: \"large language model\","
        "\"language model\", \"algorithm\", \"benchmark\", \"dataset\", \"result\", \"model family\"."
        "Repeat entities as often as they appear in facts. Avoid abbreviations in entity names. "
        "Use uniform naming of entities - if two entities have different names but mean the same thing, use one name only."
        "Output json and only json, in the following format:"
        "[{\"1\": ["
        "{\"name\":\"entity1\", \"type\":\"...\"}, {\"name\":\"entity2\", \"type\":\"...\"}"
        "]\", "
        "{\"2\":{\"name\":\"entity3\", \"type\":\"...\"}, {\"name\":\"entity1\", \"type\":\"...\"}}"
        "]"
)

entity_extraction_template2 = ("Examples of entities: \n {{ entity_samples }} "
                               "Write at least 2 and at most 5 entities that are mentioned in the text below:\n"
                               "\"{{ text }}\"\n"
                               "Output in json format as the example below. Output json and only json, lowercase:\n"
                               "["
                               "{\"name\":\"...\", \"type\":\"...\"},"
                               "{\"name\":\"...\", \"type\":\"...\"}"
                               "]\n")
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
        data = [(sql_toolkit.sql_to_element(n[0]), n[1], n[2]) for n in result]
    if len(data) > 0:
        logging.info(f"Loaded {len(data)} facts")
    return data

def _get_qualifying_abstracts(limit: int = 10) -> List[GraphElement]:
    with Session() as sess:
        query = select(SQLAElement).where(
            (SQLAElement.type_id == 'abstract') & (~SQLAElement.status.contains('entities'))
        ).order_by(SQLAElement.date.asc()).limit(limit)
        result = sess.execute(query).all()
    return [sql_toolkit.sql_to_element(n[0]) for n in result]

def _get_facts_for_abstract(abstract: GraphNode) -> List[GraphElement]:
    with Session() as sess:
        query = select(SQLAElement).where(
            (SQLAElement.type_id == 'fact') & (SQLAElement.source_id == abstract.id)
        )
        result = sess.execute(query).all()
    return [sql_toolkit.sql_to_element(n[0]) for n in result]

def _save(master_node: GraphNode, nodes: Set[GraphNode], relationships: List[GraphRelationship]):
    logging.info(f"Saving {len(nodes)} nodes and {len(relationships)} relationships")
    with sql_lock:
        logging.info("Lock acquired")
        master_node_sql = sql_toolkit.element_to_sql(master_node)
        nodes_sql = [sql_toolkit.element_to_sql(n) for n in nodes]
        relationships_sql = [sql_toolkit.element_to_sql(r) for r in relationships]
        with Session() as sess:
            with sess.begin():
                sess.merge(master_node_sql)
                for node in nodes_sql:
                    sess.merge(node)
                for rel in relationships_sql:
                    sess.merge(rel)

    logging.info(f"Saving complete")

def _get_model_response(abstract: GraphNode, facts: List[GraphNode], model: str, max_attempts: int = 5) -> List[_Entity]:
    #logging.info(f"Extracting entities for fact: \"{node.text}\"/\"{title}\"")
    template = Template(entity_extraction_template1)
    with entity_lock:
        entity_sample_objs = random.sample(entity_list, min(200, len(entity_list)))
        entity_samples = json.dumps(entity_sample_objs)
    query = template.render(facts=facts, title=abstract.title, abstract=abstract.text, entity_samples=entity_samples)
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
            with entity_lock:
                entity_list.extend(extracted_objects)
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

def _extract_from_one(abstract: GraphNode, model: str) -> Tuple[Set[GraphNode], List[GraphRelationship]]:
    global openai_api_lock
    global sql_lock
    global openai_api_usage
    facts = _get_facts_for_abstract(abstract=abstract)
    extracted_objects = _get_model_response(abstract=abstract, facts=facts, max_attempts=5, model=model)
    nodes = set()
    relationships = list()

    for entity_obj in extracted_objects:
        _extracted_object_to_nodes_and_relationship(fact_node, entity_obj, nodes, relationships)

    fact_node.status += "entities "
    logging.info(f"Created {len(nodes)} nodes and {len(relationships)} relationships")
    _save(fact_node, nodes, relationships)
    return nodes, relationships

def perform_extraction( model: str, max_count: int = 20, pool_size: int = 1):
    abstracts = _get_qualifying_abstracts(limit=max_count)
    extract_from_one_partial = partial(_extract_from_one, model=model)
    pool = ThreadPool(pool_size)
    pool.map(extract_from_one_partial, abstracts)
    pool.close()
    pool.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # model="gpt-4-0125-preview"
    model = "gpt-3.5-turbo-0125"
    # model = "gpt-4-turbo"
    perform_extraction(model=model, max_count=100, pool_size=1)
    logging.info("Entity extraction done, exiting...")


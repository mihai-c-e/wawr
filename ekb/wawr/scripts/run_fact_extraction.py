import re
from functools import partial
from typing import List, Dict, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from ekb.base.models import GraphNode, GraphRelationship

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from jinja2 import Template
from pydantic import BaseModel
from base.tools.openai_models import query_model
import logging
from multiprocessing.pool import ThreadPool
import threading
from base.tools.sql_interface import element_to_sql, Session, SQLAElement, sql_to_element
from ekb.utils import read_json

sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}

class _Entity(BaseModel):
    name: str
    type: str

class _Fact(BaseModel):
    type: str
    fact: str
    citation: str

def _get_qualifying_abstracts(limit: int = 10) -> List[GraphNode]:
    with (Session() as sess):
        stmt = select(SQLAElement).where(
            (SQLAElement.type_id == "abstract") & (~SQLAElement.status.contains('facts'))
        ).order_by(SQLAElement.date.desc()).limit(limit)
        result = sess.execute(stmt).all()
        nodes = [sql_to_element(n[0]) for n in result]
    if len(nodes) > 0:
        logging.info(f"Loaded {len(nodes)} abstracts between {nodes[-1].date} and {nodes[0].date}")
    return nodes

def _save(master_node: GraphNode, nodes: Dict[Tuple, GraphNode], relationships: List[GraphRelationship]):
    logging.info(f"Saving {len(nodes)} nodes and {len(relationships)} relationships")
    with sql_lock:
        logging.info("Lock acquired")
        master_node_sql = element_to_sql(master_node)
        nodes_sql = [element_to_sql(n) for n in nodes.values()]
        relationships_sql = [element_to_sql(r) for r in relationships]
        with Session() as sess:
            with sess.begin():
                sess.merge(master_node_sql)
                for node in nodes_sql:
                    sess.merge(node)
                sess.add_all(relationships_sql)

    logging.info(f"Saving complete")

def _get_model_response(abstract_node: GraphNode, model: str, max_attempts: int = 5) -> List[_Fact]:
    logging.info(f"Extracting facts for abstract node {abstract_node.id}/{abstract_node.title}")
    str_template = """
        Research paper title: {{ title }}
        Research paper abstract: "
        {{ abstract }}
        "
        Examine the abstract above and create a list of relevant facts,
        one sentence per fact. Assign a relevant type to each fact, such as: "introduction", "purpose", "method",
        "result", "conclusion", "achievement" or others. Assign to each fact a word-for-word comprehensive citation 
        from the abstract that justifies the fact. The citation should be large enough to provide context to the reader
        even without reading the rest of the abstract.
        Answer in json format as described below. Answer with json and only json. Escape characters appropriately
        so that your response parses as valid json when passed to Python json.loads. 
        Make sure you do not output double quotes or the escape character inside the strings. Ony use double quotes
        for json formatting.

        [{
            "type":"...", "fact":"...", "citation": "...",
        },
        ...
        ]  
        """
    template = Template(str_template)
    text = abstract_node.text.replace("\"", "'").replace("\\", "")
    query = template.render(title=abstract_node.title, abstract=text)
    attempt = 1
    while True:
        try:
            response, completion = query_model(query, model=model)
            with openai_api_lock:
                for k, v in dict(completion.usage).items():
                    openai_api_usage[k] = openai_api_usage.get(k, 0) + v
                logging.info(f"OpenAI API usage: {openai_api_usage}")
            #response = re.sub("\\\\\"", "", response)
            #response = re.sub("\\(?!\)", "", response)
           # response = response.replace("\\", "")
            response = read_json(response)
            extracted_objects = [_Fact.model_validate(r) for r in response]
            break
        except Exception as ex:
            logging.exception(ex)
            attempt += 1
            if attempt > max_attempts:
                raise
            logging.error(f"Attempt {attempt}")
    logging.info(f"Model responded with {len(extracted_objects)} facts")
    return extracted_objects

def _extracted_object_to_nodes_and_relationship(abstract_node: GraphNode, fact_obj: _Fact, nodes: Dict[Tuple, GraphNode], relationships: List[GraphRelationship]) -> None:
    # Add the fact node
    fact_meta = dict(abstract_node.meta)
    fact_meta["type"] = fact_obj.type
    fact_node = GraphNode(
        text=fact_obj.fact,
        date=abstract_node.date,
        meta=fact_meta,
        type_id="fact",
        status='',
        source_id=abstract_node.id,
        citation=fact_obj.citation,
        text_type=fact_obj.type,
        title=abstract_node.title
    )
    nodes[(fact_obj.type, fact_node.id)] = fact_node
    # Add source relationship between fact node and paper abstract
    relationships.append(GraphRelationship(
        from_node=fact_node, to_node=abstract_node, text="mentioned in", type_id="mentioned_in"
    ))
    # Add the fact type node
    fact_type_node = GraphNode(
        id=f"fact_type:{fact_obj.type}", text=fact_obj.type, type_id="fact_type"
    )
    nodes[("", fact_type_node.id)] = fact_type_node
    # Add the is_a relationship between fact type node and fact node
    relationships.append(GraphRelationship(
        from_node=fact_node, to_node=fact_type_node, text="is a", type_id="is_a"
    ))
    """for entity in fact_obj.entities:
        # Add entity node
        entity_node = GraphNode(id=f"entity:{entity.name}", text=entity.name, type_id="entity")
        nodes[(entity.type, entity_node.id)] = entity_node
        # Add relationship between entity node and fact
        relationships.append(GraphRelationship(
            from_node=entity_node, to_node=fact_node, text="mentioned in", type_id="mentioned_in"
        ))
        # Add entity type node
        entity_type_node = GraphNode(
            id=f"entity_type:{entity.type}", text=entity.type, type_id="entity_type"
        )
        nodes[("", entity_type_node.id)] = entity_type_node
        # Add the is_a relationship between entity type node and entity node
        relationships.append(GraphRelationship(
            from_node=entity_node, to_node=entity_type_node, text="is a", type_id="is_a"
        ))"""

def _extract_from_one(abstract_node: GraphNode, model: str) -> Tuple[Dict[Tuple, GraphNode], List[GraphRelationship]]:
    global openai_api_lock
    global sql_lock
    global openai_api_usage
    extracted_objects = _get_model_response(abstract_node, max_attempts=5, model=model)
    nodes = dict()
    relationships = list()

    for fact_obj in extracted_objects:
        _extracted_object_to_nodes_and_relationship(abstract_node, fact_obj, nodes, relationships)

    abstract_node.status += "facts "
    logging.info(f"Created {len(nodes)} nodes and {len(relationships)} relationships")
    _save(abstract_node, nodes, relationships)
    return nodes, relationships

def perform_extraction( model: str, max_count: int = 20, pool_size: int = 1):
    abstracts = _get_qualifying_abstracts(max_count)
    extract_from_one_partial = partial(_extract_from_one, model=model)
    pool = ThreadPool(pool_size)
    pool.map(extract_from_one_partial, abstracts)
    pool.close()
    pool.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # model="gpt-4-0125-preview"
    model = "gpt-3.5-turbo-0125"
    perform_extraction(model=model, max_count=10000, pool_size=50)
    logging.info("Fact extraction done, exiting...")


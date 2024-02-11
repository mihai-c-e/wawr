from typing import List, Dict, Tuple
from dotenv import load_dotenv
from ekb.base.models import GraphNode, GraphRelationship

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from jinja2 import Template
from pydantic import BaseModel
from ekb.wawr.models import PaperAbstract
from ekb.base.openai_models import query_model
import logging
from multiprocessing.pool import ThreadPool
import threading
from ekb.base.sql_interface import SQLABase, element_to_sql, embedding_to_sql, Session, SQLAElement, SQLARelationship, sql_to_element
from ekb.utils import read_json

sql_lock = threading.Lock()

class _Entity(BaseModel):
    name: str
    type: str

class _Fact(BaseModel):
    type: str
    fact: str
    citation: str
    entities: List[_Entity]

def _get_qualifying_abstracts(limit: int = 10) -> List[GraphNode]:
    with (Session() as sess):
        stmt = select(SQLAElement).where(
            (SQLAElement.type_id == PaperAbstract.__name__) & (~SQLAElement.status.contains('facts'))
        ).order_by(SQLAElement.date.desc()).limit(limit)
        result = sess.execute(stmt).all()
        nodes = [sql_to_element(n[0]) for n in result]
    if len(nodes) > 0:
        logging.info(f"Loaded {len(nodes)} abstracts between {nodes[-1].date} and {nodes[0].date}")
    return nodes

def _save(master_node: GraphNode, nodes: Dict[Tuple, GraphNode], relationships: List[GraphRelationship]):
    logging.info(f"Saving {len(nodes)} nodes and {len(relationships)} relationships")
    with sql_lock:
        logging.info("Lock a")
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

def _extract_from_one(abstract_node: GraphNode, max_attempts: int = 5) -> Tuple[Dict[Tuple, GraphNode], List[GraphRelationship]]:
    logging.info(f"Extracting facts for abstract node {abstract_node.id}/{abstract_node.meta['title']}")
    str_template = """
    Research paper title: {{ title }}
    Research paper abstract: "
    {{ abstract }}
    "
    Examine the abstract above and create a list of relevant facts,
    one sentence per fact. Assign a relevant type to each fact, such as: "hypothesis", "premise", "opinion",
    "contribution", "result", "achievement", "method" or others. Assign to each fact a word-for-word citation 
    from the abstract that justifies the fact. The citation should be large enough to provide context to the reader
    even without reading the rest of the abstract. For each fact, extract the entities mentioned in the fact citation 
    together with their type, such as: "model", "algorithm", "dataset" or others.  
    Answer in json format as described below. Answer with json and only json.
    
    [{
        "type":"...", "fact":"...", "citation": "...",
        "entities": [
            "name": "...",
            "type": "..."                        
        ]
    },
    ...
    ]  
    """
    template = Template(str_template)
    query = template.render(title = abstract_node.meta["title"], abstract=abstract_node.text)
    attempt = 1
    while True:
        try:
            response, completion = query_model(query, model="gpt-4-0125-preview")
            response = read_json(response)
            extracted_objects = [_Fact.model_validate(r) for r in response]
            break
        except Exception as ex:
            logging.exception(ex)
            attempt += 1
            if attempt > max_attempts:
                raise
            logging.error(f"Attempt {attempt}")
    logging.info(f"Model responded with {len(extracted_objects)} facts, creating graph")
    nodes = dict()
    relationships = list()

    for fact_obj in extracted_objects:
        # Add the fact node
        fact_meta = dict(abstract_node.meta)
        fact_meta["citation"] = fact_obj.citation
        fact_meta["type"] = fact_obj.type
        fact_node = GraphNode(
            text=fact_obj.fact,
            date=abstract_node.date,
            meta = abstract_node.meta,
            type_id="fact",
            status=''
        )
        nodes[(fact_obj.type, fact_node.id)] = fact_node
        # Add source relationship between fact node and paper abstract
        relationships.append(GraphRelationship(
            from_node=fact_node, to_node=abstract_node, text= "mentioned in", type_id= "mentioned_in"
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
        for entity in fact_obj.entities:
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
            ))
    abstract_node.status += "facts "
    logging.info(f"Created {len(nodes)} nodes and {len(relationships)} relationships")
    return nodes, relationships

def perform_extraction(max_count: int = 10, pool_size: int = 1):
    abstracts = _get_qualifying_abstracts(max_count)
    nodes, relationships = _extract_from_one(abstracts[0])
    _save(abstracts[0], nodes, relationships)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    perform_extraction(10, 1)
    logging.info("Fact extraction done, exiting...")


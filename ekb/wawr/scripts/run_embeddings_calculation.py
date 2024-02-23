from typing import List, Dict, Tuple, Type
from uuid import uuid4

from dotenv import load_dotenv
from ekb.base.models import GraphNode, GraphRelationship

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from jinja2 import Template
from pydantic import BaseModel
from ekb.wawr.models import PaperAbstract
from ekb.base.openai_models import query_model, create_embeddings
import logging
from multiprocessing.pool import ThreadPool
import threading
from ekb.base.sql_interface import SQLABase, element_to_sql, embedding_to_sql, Session, SQLAElement, SQLARelationship, \
    sql_to_element, OpenAITextEmbedding3Small, get_embedding_table_class_by_key
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
    entities: List[_Entity]

def _get_qualifying_nodes(embeddings_table: Type = OpenAITextEmbedding3Small, limit: int = 10) -> List[GraphNode]:
    with (Session() as sess):
        stmt = select(SQLAElement).join(
            embeddings_table, isouter=True, onclause= (SQLAElement.id == embeddings_table.node_id)
        ).where(
            (embeddings_table.node_id == None) & (SQLAElement.record_type=='node')
        ).limit(limit)
        result = sess.execute(stmt).all()
        nodes = [sql_to_element(n[0]) for n in result]
    if len(nodes) > 0:
        logging.info(f"Loaded {len(nodes)} nodes")
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

def perform_embedding(max_count: int, batch_size: int, embedding_key: str):
    nodes = _get_qualifying_nodes(limit=max_count)
    logging.info(f"Calculating embeddings for {len(nodes)} nodes")
    embeddings_table = get_embedding_table_class_by_key(embedding_key)
    for i in range(0, len(nodes), batch_size):
        logging.info(f"Batch {i} - {i+batch_size} of {len(nodes)}")
        data = [n.text for n in nodes[i:i+batch_size]]
        logging.info("Creating embeddings")
        embeddings = create_embeddings(data=data, model="text-embedding-3-small")
        logging.info("Updating elements")
        sql_objects: List[embeddings_table] = list()
        for node, emb in zip(nodes[i:i+batch_size], embeddings):
            node.embeddings[embedding_key] = emb
            emb_row = embedding_to_sql(node, embedding_key)
            sql_objects.append(emb_row)
        logging.info("Saving to database")
        with Session() as sess:
            with sess.begin():
                sess.add_all(sql_objects)
        logging.info("Saved to database")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    perform_embedding(100000, 1000, embedding_key="text-embedding-3-small")
    logging.info("Embedding calculation done, exiting...")


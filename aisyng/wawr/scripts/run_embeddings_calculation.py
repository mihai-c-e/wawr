from typing import List, Dict, Tuple, Type
from uuid import uuid4

from dotenv import load_dotenv
from aisyng.base.models import GraphNode, GraphRelationship

load_dotenv('../../../wawr_ingestion.env')

from sqlalchemy import select
from pydantic import BaseModel
from base.tools.openai_models import create_embeddings
import logging
import threading
from base.tools.sql_interface import Session, SQLAElement, SQLToolkit
from aisyng.wawr.wawr_embeddings import embedding_pool, Embedder

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

def _get_qualifying_nodes(embedder: Embedder, limit: int, sql_toolkit: SQLToolkit) -> List[GraphNode]:
    embeddings_table = embedder.table
    with (Session() as sess):
        stmt = select(SQLAElement).join(
            embeddings_table, isouter=True, onclause= (SQLAElement.id == embeddings_table.node_id)
        ).where(
            (embeddings_table.node_id == None) & (SQLAElement.record_type=='node')
        ).limit(limit)
        result = sess.execute(stmt).all()
        nodes = [sql_toolkit.sql_to_element(n[0]) for n in result]
    if len(nodes) > 0:
        logging.info(f"Loaded {len(nodes)} nodes")
    return nodes

def perform_embedding(max_count: int, batch_size: int, embedder: Embedder, sql_toolkit: SQLToolkit):
    nodes = _get_qualifying_nodes(embedder=embedder, limit=max_count, sql_toolkit=sql_toolkit)
    logging.info(f"Calculating embeddings for {len(nodes)} nodes")
    for i in range(0, len(nodes), batch_size):
        logging.info(f"Batch {i} - {i+batch_size} of {len(nodes)}")
        data = [n.text for n in nodes[i:i+batch_size]]
        logging.info("Creating embeddings")
        embeddings = embedder.create_embeddings(data=data)
        logging.info("Updating elements")
        sql_objects: List[embedder.table] = list()
        for node, emb in zip(nodes[i:i+batch_size], embeddings):
            node.embeddings[embedder.name] = emb
            emb_row = sql_toolkit.embedding_to_sql(node, embedding_key=embedder.name)
            sql_objects.append(emb_row)
        logging.info("Saving to database")
        with Session() as sess:
            with sess.begin():
                sess.add_all(sql_objects)
        logging.info("Saved to database")


if __name__ == '__main__':

    sql_toolkit = SQLToolkit(embedding_pool=embedding_pool)
    logging.basicConfig(level=logging.INFO)
    embedder: Embedder = embedding_pool.get_embedder(embedding_key="text-embedding-3-small")
    perform_embedding(1000000, 2000, embedder=embedder, sql_toolkit=sql_toolkit)
    logging.info("Embedding calculation done, exiting...")


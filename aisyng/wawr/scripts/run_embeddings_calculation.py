from typing import List
from uuid import uuid4
import logging
import threading
from dotenv import load_dotenv

load_dotenv('../../../wawr_ingestion.env')

from aisyng.wawr.wawr_embeddings import Embedder
from aisyng.wawr.context import WAWRContext

sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}
wawr_context: WAWRContext = WAWRContext.create_default()



def perform_embedding(max_count: int, batch_size: int, embedder: Embedder):
    nodes = wawr_context.get_persistence().get_nodes_without_embeddings(embedding_key=embedder.name, limit=max_count)
    logging.info(f"Calculating embeddings for {len(nodes)} nodes")
    for i in range(0, len(nodes), batch_size):

        logging.info(f"Batch {i} - {i+batch_size} of {len(nodes)}")
        nodes_batch = nodes[i:i+batch_size]
        data = [n.text if n.text != '' else 'none' for n in nodes_batch]

        logging.info("Creating embeddings")
        embeddings = embedder.create_embeddings(data=data)

        logging.info("Updating elements")
        sql_objects: List[embedder.table] = list()
        for node, emb in zip(nodes_batch, embeddings):
            node.embeddings[embedder.name] = emb

        logging.info("Saving to database")
        wawr_context.get_persistence().sqli.persist_embeddings(
            nodes = nodes_batch,
            embedding_key=embedder.name,
            batch_size=batch_size
        )

        logging.info("Saved to database")


if __name__ == '__main__':

    logging.getLogger().setLevel(level=logging.INFO)
    embedder: Embedder = wawr_context.get_embedding_pool().get_embedder(embedding_key="text-embedding-3-small-128")
    perform_embedding(300000, 2000, embedder=embedder)
    logging.info("Embedding calculation done, exiting...")

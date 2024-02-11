from ekb.llm_interface import ChatGPTConnection, EmbeddingService
from ekb.db_interface import Neo4jConnection
import os
from langchain.vectorstores import SKLearnVectorStore
from langchain.embeddings import OpenAIEmbeddings


def generate_node_keywords():
    db = Neo4jConnection()

    node_keywords = db.get_all_node_keywords()
    to_embed = set()
    for v in node_keywords.values():
        to_embed.update(v)
    to_embed = list(to_embed)
    embedding_func = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API'])

    vector_store = SKLearnVectorStore.from_texts(
        texts=to_embed,
        embedding=embedding_func,
        persist_path="../data/sklearn_node_keywords",  # persist_path and serializer are optional
        serializer="json",
    )
    vector_store.persist()

def generate_relationship_keywords():
    db = Neo4jConnection()
    rel_keywords = db.get_all_relation_keywords()
    to_embed = rel_keywords

    embedding_func = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API'])
    vector_store = SKLearnVectorStore.from_texts(
        texts=to_embed,
        embedding=embedding_func,
        persist_path="../data/sklearn_relationship_keywords",  # persist_path and serializer are optional
        serializer="json",
    )
    vector_store.persist()

if __name__ == '__main__':
    generate_node_keywords()
    generate_relationship_keywords()



    #embeddings = es.get_embeddings(to_embed)
    #result = pd.DataFrame(zip(to_embed, embeddings), columns=['text', 'vector'])
    #result = result.set_index('text')
    #result.to_csv('../data/embeddings.csv')

from dotenv import load_dotenv
load_dotenv('../wawr_ingestion.env')

from base.tools.openai_models import create_embeddings
from base.tools.sql_interface import SQLAElement

def test_embed_and_find_by_similarity():
    data = ["what do we know about gpt-4's performance?"]
    embedding_key = "text-embedding-3-small"
    embeddings = create_embeddings(data=data, model="text-embedding-3-small")[0]
    result = SQLAElement.find_by_similarity(with_string=data[0], with_vector=embeddings, distance_threshold=100, embedding_key=embedding_key, limit=50)


if __name__ == '__main__':
    results = test_embed_and_find_by_similarity()
    print(results)
from dotenv import load_dotenv
load_dotenv('../wawr_ingestion.env')

from base.tools.openai_models import create_embeddings
from base.tools.sql_interface import SQLAElement

def test_embed_and_find_by_similarity():
    data = ["GPT 4", "chess"]
    embedding_key = "text-embedding-3-small"
    embeddings = create_embeddings(data=data, model="text-embedding-3-small")
    result = SQLAElement.find_by_similarity(with_strings=data, with_vectors=embeddings, distance_threshold=100,
                                            embedding_key=embedding_key, limit=50, type_ids=["Fact", "PaperAbstract"])
    return result


if __name__ == '__main__':
    results = test_embed_and_find_by_similarity()
    results = [(r[0], r[1]) for r in results]
    print(results)
import logging

from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')
from ekb.base.topic import TopicMeta
from ekb.base.tools.default_toolkit import topic_solver_v1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    question = "Is GPT 3.5 better than GPT 4?"
    question = "Can language models perform reasoning?"
    question = "What the fuck is chat gpt?"
    meta = TopicMeta(source_id="test", embedding_key="text-embedding-3-small", model="gpt-4-0125-preview", distance_threshold=0.35, limit=1000)
    topic_solver_v1(topic=question, meta=meta, in_thread=True)
    print("Done")
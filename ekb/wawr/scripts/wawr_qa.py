import logging

from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')
from ekb.base.topic import TopicMeta
from ekb.base.tools.default_toolkit import topic_solver_v1, topic_solver_v2, break_down_question


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    question = "Is GPT 3.5 better than GPT 4?"
    question = "Can language models perform reasoning?"
    question = "Find me arguments for language models being sentient"
    question = "What games can language models play?"
    # question = "What language models have been tested on chess?"
    # question = "What language models play starcraft?"
    question = "How well do language models play chess?"
    meta = TopicMeta(source_id="remove", embedding_key="text-embedding-3-small", model="gpt-4-0125-preview", distance_threshold=0.5, limit=300)
    topic_solver_v2(topic=question, meta=meta, in_thread=True)
    #test=break_down_question(question, "gpt-4-0125-preview")
    print("Done")
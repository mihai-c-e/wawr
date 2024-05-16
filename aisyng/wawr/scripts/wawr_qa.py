from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')

import datetime
import logging
from wawr.models.topic import TopicMeta
from aisyng.base.tools.default_toolkit import EKBToolkit
from aisyng.wawr.context import WAWRContext
from aisyng.wawr.models.topic import TopicNode, DirectSimilarityTopicSolver
from aisyng.wawr.models.models_factory import create_topic_node

wawr_context: WAWRContext = WAWRContext.create_default()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    question = "Is GPT 3.5 better than GPT 4?"
    question = "Can language models perform reasoning?"
    question = "Find me arguments for language models being sentient"
    question = "What games can language models play?"
    # question = "What language models have been tested on chess?"
    # question = "What language models play starcraft?"
    question = "How well do language models play chess?"
    embedding_key = "text-embedding-3-small-128"
    #toolkit = EKBToolkit(embedding_pool=wawr_context.get_embedding_pool())
    #meta = TopicMeta(source_id="remove", embedding_key=embedding_key, model="gpt-4-0125-preview", distance_threshold=0.5, limit=300)
    topic_node = create_topic_node(ask=question, source_id="remove")
    topic_solver = DirectSimilarityTopicSolver(
        from_date=datetime.datetime(2000, 3, 1),
        model="gpt-4-turbo",
        distance_threshold=0.5,
        embedding_key=embedding_key,
        limit=100
    )
    topic_solver.solve
    answer = toolkit.topic_solver_v2(embedding_key=embedding_key, topic=question, meta=meta, in_thread=False)
    #test=break_down_question(question, "gpt-4-0125-preview")
    print("Done")
from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')

import datetime
import logging
from aisyng.wawr.context import WAWRContext
from aisyng.wawr.models.payload import DirectSimilarityTopicSolver
from aisyng.wawr.models.models_factory import create_topic_node
from aisyng.base.llms.base import LLMName
from aisyng.wawr.workers import solve_topic, init_topic_solving

wawr_context: WAWRContext = WAWRContext.create_default()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    question = "Is GPT 3.5 better than GPT 4?"
    # question = "Can language models perform reasoning?"
    question = "Find arguments for language models being sentient"
    question = "What games can language models play?"
    question = "What language models have been tested on chess?"
    question = "what is gpt-4?"
    question = "Does GPT-3.5 outperform GPT-4 on any topics?"

    # question = "What language models play starcraft?"
    #question = "How well do language models play chess?"
    embedding_key = "text-embedding-3-small-128"
    topic_node = create_topic_node(ask=question, source_id="remove")
    topic_solver = DirectSimilarityTopicSolver(
        from_date=datetime.datetime(2000, 3, 1),
        model="gpt-4-turbo",
        distance_threshold=0.7,
        embedding_key=embedding_key,
        limit=200,
        llm_name = LLMName.OPENAI_GPT_4_TURBO
    )
    topic_solver_node = init_topic_solving(topic_node=topic_node, context=wawr_context, topic_solver=topic_solver)
    solve_topic(topic_node=topic_node, topic_solver_node=topic_solver_node, context=wawr_context)

    #answer = toolkit.topic_solver_v2(embedding_key=embedding_key, topic=question, meta=meta, in_thread=False)
    #test=break_down_question(question, "gpt-4-0125-preview")
    print("Done")
from typing import Tuple, Dict

import pandas as pd
from datetime import datetime
import time
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings

from ekb.utils import read_json, read_html


from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from neo4j import GraphDatabase
from py2neo.data import Relationship, Node
from py2neo import Graph, NodeMatcher, RelationshipMatcher

from ekb.utils import check_env
import logging

class ChatGPTConnection:
    model_name: str
    def __init__(self, model_name: str = 'gpt-3.5-turbo-16k-0613'):
        self.cllm = None
        self.memory = None
        self.model_name = model_name
        self.connect(model_name=model_name)
        pass

    def check_env(self):
        required_keys = {'OPENAI_API'}
        check_env(required_keys)

    def query_model(self, query: str, max_attempts: int = 5, temperature: float=0.1) -> Tuple[str, Dict]:
        attempt = 1
        if isinstance(query, str):
            query = [HumanMessage(content=query)]
        while True:
            try:
                response = self.cllm(query, temperature=temperature)
                return response.content, response
            except Exception as ex:
                logging.exception(ex)
                attempt += 1
                if attempt > max_attempts:
                    logging.error(f"Server did not reply after {max_attempts} attempts, stopping...")
                    raise
                self.connect()
                logging.error(f"Attempt {attempt}")



    def connect(self, model_name: str = 'gpt-3.5-turbo-16k-0613'):
        self.check_env()
        self.cllm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API'], model_name=model_name)
        self.memory = ConversationBufferMemory()



    def question_to_keywords(self, question: str):
        messages = [
            SystemMessage(
                content='You are an agent that receives a free text questions and, instead of trying to answer it, '
                        'answer with the most relevant nodes and relations that would put the question into a '
                        'knowledge graph, in json format. Make sure to '
                        'include verbs in relations. Example: for "How do I reduce model hallucinations?" you should '
                        'answer with: "{"nodes":["model", "hallucination"], "relations":["application", "apply to"]}. '
                        'Answer with a valid json and only with '
                        'a valid json.'
            ),
            HumanMessage(
                content=question
            ),
        ]
        response = self.cllm(messages, temperature=0.1).content
        response=read_json(response)
        return response


    def match_against_keywords(self, set1, set2) -> dict:
        system_content = """
                    You are an agent that receives 2 sets of keywords and matches the first set with zero or more similar keywords from the second set.
                    For instance, if you receive the prompt "('duck', 'machine learning', 'bread'), ('bird', 'mouse', 'linear regression', 
                    'neural network')" you should answer with: "{'duck':['bird'],'machine learning':['linear regression', 'neural 
                    network']}. If there are no matches of high similarity, answer with {}. Answer in JSON format.
                """
        messages = [
            SystemMessage(
                content=''
            ),
            HumanMessage(
                content=f'{system_content}\nConsidering the rules above, match the following keywords: {tuple(set1)}\n against these:{tuple(set2)}.'
            ),
        ]
        response = self.cllm(messages, temperature=0.1).content
        response = read_json(response)
        return response

    def split_match_against_keywords(self, set1, set2, chunk_size: int = 1000) -> dict:
        set2 = list(set2)
        to_ret = dict()
        for i in range(0, len(set2), chunk_size):
            chunk_result = self.match_against_keywords(set1, set2[i:i+chunk_size])
            key_set = set(chunk_result.keys()).union(to_ret.keys())
            to_ret = {k:to_ret.get(k, []) + chunk_result.get(k, []) for k in key_set}
        return to_ret

    def match_against_dict(self, set1: set, dict2: dict, chunk_size: int = 1000) -> dict:
        to_ret = {k:list() for k in dict2.keys()}
        for k in dict2.keys():
            logging.info(f'Matching keywords against {k}:')
            to_ret[k] = self.split_match_against_keywords(set1, dict2[k], chunk_size=chunk_size)
            logging.info(to_ret[k])

    def get_similar_graph_members(self, suggested_elements, labels, names, relations, properties):
        pass

    def compose_answer_from_paths(self, question, paths, refs):
        paths_string = '\n'.join([f'{i+1}. ' + '-'.join(path) for i, path in enumerate(paths)])
        content = (f"The following are citations from research papers:\n"
                   f"{refs}.\n\nThis is the question: {question}"
                   f" Using only the citations above, try to infer and argument an answer to the question."
                   f" Include index references to citations and exclude citations that are not clearly related to the question.\n"
                   f"      "
                   #f"Insert index references to the corresponding facts in your answer."
                   )
        messages = [
            SystemMessage(
                content=''
            ),
            HumanMessage(
                content=content
            ),
        ]
        response = self.cllm(messages, temperature=0.1).content

        return response

    def compose_answer_from_paths_news_style(self, question, paths, refs, style: str = 'journalistic', temperature: float = 0.1):
        paths_string = '\n'.join([f'{i+1}. ' + '-'.join(path) for i, path in enumerate(paths)])
        content = (f"The following are citations from research papers:\n"
                   f"{refs}.\n\n This is the question: {question}"
                   f" Using only the citations above, try to infer an answer to the question. "
                   f"Write in {style} style as for a person not familiar with the topic."
                   f"Be thorough and try to address the references individually as much as possible. Then, "
                   f"output a section starting with \"The author's view is ... \" where you make an inference on what "
                   f"the next important related research would be. "
                   f" Include index references to citations as provided, but do not list references in your response.\n"
                   #f"Insert index references to the corresponding facts in your answer."
                   )
        content = (f"The following are paths in a neo4j knowledge graph, containing nodes, relationships and citations extracted from research papers:\n"
                   f"{refs}.\n\n This is the question: {question}"
                   f" Using only the paths above, try to infer an answer to the question. If none of the paths are "
                   f"relevant to the question answer, say so and do not provide any other answer."
                   f"If some or all paths are relevant to the question, write the answer in {style} style as for a "
                   f"person not familiar with the topic."
                   f"Be thorough and try to address the references individually as much as possible. Then, "
                   f"output a section starting with \"The author's view is ... \" where you make an inference on what "
                   f"the next important related research would be. "
                   f" Include index references to citations as provided, referencing the reference_id field, but do not list references in your response.\n"
                   f" Provide your answer as a html snipped that will be embedded into an existing html page with an "
                   f"existing title. Use "
                   f"html "
                   f"to format your answer - use paragraphs, styling and whatever else you consider necessary."
                   # f"Insert index references to the corresponding facts in your answer."
                   )
        messages = [
            SystemMessage(
                content=''
            ),
            HumanMessage(
                content=content
            ),
        ]
        response = self.cllm(messages, temperature=temperature).content
        response = read_html(response)
        return response


class KGPromptWriter:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.memory = ConversationBufferMemory()

    def act(self, document: str, prompt: str, result: str, feedback: str):
        message = f'Paper abstract:\n"{document}"\n'
        if prompt is not None:
            message += f'\nYour previous instructions:\n"{prompt}"\n'
        if result is not None:
            message += f'\nRecomposed abstract:\n"{result}"\n'
        if feedback is not None:
            message += f'\nFeedback:\n"{prompt}"\n'
        message += 'Write a new set of instructions to help extract a more complete knowledge graph, according to the feedback:\n '
        messages = [
            SystemMessage(
                content='Your goal is to write and refine instructions for an AI system, asking it to decompose a research abstract into'
                        'a knowledge graph. You receive the abstract, your previous instructions, an abstract recomposed '
                        'based on the knowledge graph and feedback about the similarity of the 2.'
            ),
            HumanMessage(
                content= message
            ),
        ]
        response = self.model(messages, temperature=0.1).content
        return response

class KGGenerator:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.memory = ConversationBufferMemory()

    def act(self, document: str, prompt: str):
        message = f'Paper abstract:\n"{document}"\n'
        messages = [
            SystemMessage(
                content=prompt
            ),
            HumanMessage(
                content= message
            ),
        ]
        response = self.model(messages, temperature=0.1).content
        return response

class KGRecomposer:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.memory = ConversationBufferMemory()

    def act(self, document: str):
        message = f'Knowledge  graph:\n"{document}"\n'
        messages = [
            SystemMessage(
                content="The following is a knowledge graph extracted from a paper abstract. Try to recompose the abstract."
            ),
            HumanMessage(
                content= message
            ),
        ]
        response = self.model(messages, temperature=0.1).content
        return response

class KGFeedback:
    def __init__(self, model: BaseChatModel):
        self.model = model
        self.memory = ConversationBufferMemory()

    def act(self, document: str, recomposition: str):
        messages = [
            SystemMessage(
                content='These are 2 research paper abstracts. Provide a similarity score between them and point out '
                        'what does not match or what is missing from the second abstract when compared to the first:'
            ),
            HumanMessage(
                content= f'\nAbstract 1:"{document}".\n\nAbstract 2:"{recomposition}"'
            ),
        ]
        response = self.model(messages, temperature=0.1).content
        return response


class EmbeddingService:
    def __init__(self):
        self.model = None
        self.connect()
        pass

    def check_env(self):
        required_keys = {'OPENAI_API'}
        check_env(required_keys)

    def connect(self):
        self.check_env()
        self.model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ['OPENAI_API'])

    def get_embeddings(self, s: list):
        embedded = self.model.embed_documents(s)
        return embedded


import os
import logging
import json
from typing import List

import pandas as pd

from aisyng.base.models import GraphNode, GraphElement, GraphRelationship
from jinja2 import Template
from nltk.tokenize import sent_tokenize

from llm_interface import ChatGPTConnection




def extract_facts_from_one(source: GraphNode, prompt: str = None):
    if prompt is None:
        prompt = """
        Paper title: {{ title }}
        Paper abstract: {{ abstract }}
        
        List the relevant facts from the paper abstract above, one fact per row, in json format as:
        [
            {"fact":"...", "citation":"...", "fact_type":"..."},
            {"fact":"...", "citation":"...", "fact_type":"..."},
        ]
        where:
         - citation is a word-for-word extract from the abstract, justifying the fact
         - fact_type is a category telling if the fact is: hypothesis, contribution, result or other relevant category.
        Output json and only json.
        
           
        """
    prompt = Template(prompt).render(title=source.meta["title"], abstract=source.text)
    llm = ChatGPTConnection(model_name="gpt-4-0125-preview")
    response = llm.query_model(prompt)
    print(response)
    return

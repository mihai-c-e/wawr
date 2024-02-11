import os
import logging
import json
from typing import List

import pandas as pd

from base.models import GraphNode
from ekb.wawr.models import PaperAbstract
from jinja2 import Template
from nltk.tokenize import sent_tokenize

from llm_interface import ChatGPTConnection


def _prepare_ingested_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preparing data")
    df['date'] = pd.to_datetime(df['versions'].apply(lambda x: x[0]['created']), format="%a, %d %b %Y %H:%M:%S GMT")
    df['lm'] = df['abstract'].str.lower().str.contains('language model')
    df['abstract'] = df['abstract'].str.replace('\n', ' ')
    return df

def _filter_ingested_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['lm']) & (df["date"].dt.year >= 2014)]
    return df

def load_json_source(source_file: str = None):
    source_file = source_file or os.environ["ARXIV_JSON_FILE"]
    if not source_file:
        raise ValueError("Source file not found")

    logging.info(f"Reading source file at: {source_file}")
    data = []
    with open(source_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    df = _prepare_ingested_data(df)
    df = _filter_ingested_data(df)
    logging.info(f"Read {df.shape[0]} records from source.")
    return df

def ingested_data_to_nodes(df: pd.DataFrame) -> List[PaperAbstract]:
    if df.shape[0] == 0:
        return list()
    return df[~df['id'].isna()].apply(lambda x: PaperAbstract.from_series(x), axis=1).to_list()

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
    prompt = Template(prompt).render(title = source.meta["title"], abstract = source.text)
    llm = ChatGPTConnection(model_name="gpt-4-0125-preview")
    response = llm.query_model(prompt)
    print(response)
    return


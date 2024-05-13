import logging
import os
import json
import pandas as pd
from typing import List, Dict, Any
from aisyng.wawr.models import PaperAbstract
from aisyng.base.models import GraphElement, GraphNode, GraphRelationship
from aisyng.wawr.model_factory import create_abstract_node, create_title_node_from_abstract_info, \
    create_title_to_abstract_relationship


def _prepare_ingested_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preparing data")
    df['date'] = pd.to_datetime(df['versions'].apply(lambda x: x[0]['created']), format="%a, %d %b %Y %H:%M:%S GMT")
    df['lm'] = df['abstract'].str.lower().str.contains('language model')
    df['abstract'] = df['abstract'].str.replace('\n', ' ')
    df['id'] = df['id'].astype(str)
    df=df.fillna("")
    df = df[(df['lm']) & (df["date"].dt.year >= 2014)]
    return df


def load_json_source(source_file: str = None) -> pd.DataFrame:
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
    logging.info(f"Read {df.shape[0]} records from source.")
    return df


def ingested_data_to_nodes(df: pd.DataFrame) -> List[GraphElement]:
    if df.shape[0] == 0:
        return list()
    node_info_list = df[~df['id'].isna()].apply(lambda x: PaperAbstract.model_validate(x.to_dict()), axis=1).to_list()
    abstract_nodes = [create_abstract_node(ni) for ni in node_info_list]
    title_nodes = list()
    abstract_title_relationships = list()
    for abstract_node in abstract_nodes:
        title_node = create_title_node_from_abstract_info(abstract_node)
        title_nodes.append(title_node)
        abstract_title_relationships.append(create_title_to_abstract_relationship(abstract_node, title_node))
    return abstract_nodes + title_nodes + abstract_title_relationships

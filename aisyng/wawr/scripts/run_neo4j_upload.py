import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')

import logging
from base.tools.sql_interface import Session
from neo4j import GraphDatabase
from jinja2 import Template

import pandas as pd


def get_neo4j_connection():
    URI = os.environ["AURA_CONNECTION_URI"]
    AUTH = (os.environ["AURA_USERNAME"], os.environ["AURA_PASSWORD"])

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
    return driver

def load_db_relationships():
    logging.info("Loading relationships...")
    relationships = pd.read_sql(sql="SELECT * FROM public.relationships a left join elements b on a.id=b.id",
                                con=Session().get_bind())
    relationships["date"] = pd.to_datetime(relationships["date"])
    relationships["created_date"] = pd.to_datetime(relationships["created_date"])
    logging.info(f"Loaded {relationships.shape[0]} relationships")
    return relationships

def datetime_to_int(d: datetime) -> int:
    if d is None or d is pd.NaT:
        return -1
    return int(d.strftime("%Y%M%d"))

def upload_n4j_relationships(relationships: pd.DataFrame):
    driver = get_neo4j_connection()
    query_template = Template(
        "MATCH (n1 {id:\"{{ id1 }}\" }) "
        "MATCH (n2 {id:\"{{ id2 }}\" }) "
        "CREATE (n1)-[:{{ rel_name }} {id:'{{ id }}', date:{{ date }}, "
        "created_date:{{ created_date }}, status:'{{ status }}' }]->(n2)"
    )
    for i, row in relationships.iterrows():
        summary = driver.execute_query(
            query_template.render(
                id1=row["from_node_id"].replace("\\", "\\\\").replace("\"", "\\\""),
                id2=row["to_node_id"].replace("\\", "\\\\").replace("\"", "\\\""),
                rel_name=row["text"].replace(" ", "_").upper(),
                id=row["id"].iloc[0], date=datetime_to_int(row["date"]),
                created_date=datetime_to_int(row["created_date"]), status=row["status"]
            ),
            database_="neo4j",
        ).summary
        if i % 100 == 99:
            logging.info(f"{i}/{relationships.shape[0]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    relationships = load_db_relationships()
    upload_n4j_relationships(relationships)
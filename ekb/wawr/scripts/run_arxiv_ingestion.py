from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')
from sqlalchemy import select

from wawr.models import PaperAbstract

import logging
import pandas as pd
from ekb.wawr.scripts.workers import load_json_source, ingested_data_to_nodes, extract_facts_from_one
from ekb.base.sql_interface import SQLABase, element_to_sql, embedding_to_sql, Session, SQLAElement

def get_last_ingested_abstract_id():
    with Session() as sess:
        stmt = select(SQLAElement).where(
            SQLAElement.type_id == PaperAbstract.__name__
        ).order_by(SQLAElement.id.desc()).limit(1)
        result = sess.execute(stmt).fetchone()
    logging.info(f"Last abstract id in database: {'none' if result is None else result[0].id}")
    return None if result is None else result[0].id

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    last_abstract_id = get_last_ingested_abstract_id()
    data = load_json_source()
    logging.info(f"Loaded {len(data)} records from raw ingestion file")
    if last_abstract_id is not None:
        data = data[data["id"] > last_abstract_id]
    logging.info(f"Turning {len(data)} records into nodes")
    pa_nodes = ingested_data_to_nodes(data)
    logging.info(f"Saving {len(pa_nodes)} records to database")
    sql_nodes = {x.id: element_to_sql(x) for x in pa_nodes}
    with Session() as sess:
        with sess.begin():
            sess.begin()
            sess.add_all(list(sql_nodes.values()))
            sess.commit()


from dotenv import load_dotenv

load_dotenv('wawr_ingestion.env')

import logging
from aisyng.wawr.ingestion import ingested_data_to_nodes, load_json_source
from aisyng.wawr.context import WAWRContext



if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    wawr_context = WAWRContext.create_default()

    last_abstract_id = wawr_context.get_persistence().get_last_ingested_abstract_id()
    data = load_json_source()
    logging.info(f"Loaded {len(data)} records from raw ingestion file")
    # 2404.16829
    if last_abstract_id is not None:
        data = data[data["id"] > last_abstract_id]

    logging.info(f"Turning {len(data)} records into nodes")
    pa_nodes = ingested_data_to_nodes(data)
    logging.info(f"Saving {len(pa_nodes)} records to database")
    wawr_context.get_persistence().sqli.persist(objects_add=pa_nodes)


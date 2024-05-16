import logging
from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')
from aisyng.wawr.context import WAWRContext

wawr_context: WAWRContext = WAWRContext.create_default()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rels = wawr_context.get_persistence().get_all_fact_and_fact_type_relationships(limit=100000000)
    wawr_context.get_persistence().neo4ji.persist(rels)
    logging.info("Fact extraction done, exiting...")


import logging
from dotenv import load_dotenv
load_dotenv('../../../wawr_ingestion.env')
from aisyng.wawr.context import WAWRContext

wawr_context: WAWRContext = WAWRContext.create_default()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    facts = wawr_context.get_persistence().get_all_facts_and_fact_types(limit=100000000)
    wawr_context.get_persistence().neo4ji.persist(facts)
    logging.info("Fact extraction done, exiting...")


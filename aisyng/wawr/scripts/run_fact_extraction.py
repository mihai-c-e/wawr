import json
from functools import partial
from typing import List, Dict, Tuple
from uuid import uuid4
from jinja2 import Template
import logging
from multiprocessing.pool import ThreadPool
import threading
import traceback


from dotenv import load_dotenv
from aisyng.base.models.graph import GraphNode, GraphRelationship, GraphElement

load_dotenv('../../../wawr_ingestion.env')


from aisyng.utils import read_json
from aisyng.wawr.context import WAWRContext
from aisyng.wawr.models.graph import Fact
from aisyng.wawr.models.graph import WAWRGraphElementTypes
from aisyng.wawr.scripts._fact_extraction_prompts import fact_extraction_prompt_1
from aisyng.base.llms.base import LLMName
# from base.llms.openai_models import query_model


sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}
wawr_context: WAWRContext = WAWRContext.create_default()
errors = list()



def _get_model_response(abstract_node: GraphNode, model_name: LLMName, max_attempts: int = 5) -> List[Fact]:
    logging.info(f"Extracting facts for abstract node {abstract_node.id}/{abstract_node.title}")
    template = Template(fact_extraction_prompt_1)
    text = (abstract_node.text.replace("\"", "").replace("'", "").
            replace("`", "").replace("\\", ""))
    title = abstract_node.title.replace("\"", "")
    query = template.render(title=title, abstract=text)
    attempt = 1
    while True:
        try:
            response, completion = (wawr_context.llm_providers.
                                    get_by_model_name(model_name=model_name).
                                    query_model(query, model=model_name))
            response = response.replace("\\", "")
            with openai_api_lock:
                for k, v in dict(completion.usage).items():
                    openai_api_usage[k] = openai_api_usage.get(k, 0) + v
                logging.info(f"OpenAI API usage: {openai_api_usage}")

            response = read_json(response)
            extracted_objects = [Fact.model_validate_with_date(r, abstract_node.date) for r in response]
            break
        except Exception as ex:
            logging.exception(ex)
            attempt += 1
            if attempt > max_attempts:
                raise
            logging.error(f"Attempt {attempt}")
    logging.info(f"Model responded with {len(extracted_objects)} facts")
    return extracted_objects

def create_fact_id(fact: Fact, abstract_node: GraphNode, index: int) -> str:
    return f"{abstract_node.id}.{fact.type}.{index}"

def _extracted_object_to_nodes_and_relationship(
        abstract_node: GraphNode,
        fact_obj: Fact,
        index: int,
        nodes: Dict[Tuple, GraphNode],
        relationships: List[GraphRelationship]
) -> None:
    # Add the fact node
    fact_node = GraphNode(
        id=create_fact_id(fact=fact_obj, abstract_node=abstract_node, index=index),
        text=fact_obj.text,
        date=fact_obj.date,
        meta=fact_obj,
        type_id=WAWRGraphElementTypes.Fact,
        status='',
        source_id=abstract_node.id,
        citation=fact_obj.citation,
        text_type=fact_obj.type,
        title=abstract_node.title
    )
    nodes[(fact_obj.type, fact_node.id)] = fact_node
    # Add source relationship between fact node and paper abstract
    relationships.append(GraphRelationship(
        from_node=fact_node,
        to_node=abstract_node,
        text=WAWRGraphElementTypes.IsExtractedFrom,
        type_id=WAWRGraphElementTypes.IsExtractedFrom
    ))
    # Add the fact type node
    fact_type_node = GraphNode(
        id=f"fact_type:{fact_obj.type}",
        text=fact_obj.type,
        type_id=WAWRGraphElementTypes.FactType
    )
    nodes[("", fact_type_node.id)] = fact_type_node
    # Add the is_a relationship between fact type node and fact node
    relationships.append(GraphRelationship(
        from_node=fact_node,
        to_node=fact_type_node,
        text=WAWRGraphElementTypes.IsA,
        type_id=WAWRGraphElementTypes.IsA
    ))

def _extract_from_one(abstract_node: GraphNode, model_name: LLMName) -> List[GraphElement]:
    global errors
    logging.info(f"Extracting facts from {abstract_node.id}")
    try:
        global openai_api_lock
        global sql_lock
        global openai_api_usage
        extracted_objects = _get_model_response(abstract_node, max_attempts=5, model_name=model_name)
        nodes = dict()
        relationships = list()

        for i, fact_obj in enumerate(extracted_objects):
            _extracted_object_to_nodes_and_relationship(
                abstract_node=abstract_node,
                fact_obj=fact_obj,
                index=i,
                nodes=nodes,
                relationships=relationships
            )

        nodes_list = list(nodes.values())
        to_add = nodes_list + relationships

        logging.info(f"Created {len(nodes)} nodes and {len(relationships)} relationships")
        #with sql_lock:
        #    wawr_context.get_persistence().persist(nodes_list+relationships)
        return to_add
    except Exception as ex:
        logging.exception(ex)
        error_info = {'abstract': abstract_node.id, 'message': traceback.format_exception_only()}
        errors.append(error_info)
        raise

def perform_extraction( model_name: LLMName, max_count: int = 20, pool_size: int = 1, batch_size: int=100):
    abstracts = wawr_context.get_persistence().get_abstracts_without_facts(max_count)
    extract_from_one_partial = partial(_extract_from_one, model_name=model_name)
    pool = ThreadPool(pool_size)
    for i in range(0, len(abstracts), batch_size):
        results: List[List[GraphElement]] = pool.map(extract_from_one_partial, abstracts[i:i+batch_size])
        to_add = [ge for ge_list in results for ge in ge_list]
        logging.info(f"Saving {len(to_add)} objects to database")
        with sql_lock:
            wawr_context.get_persistence().sqli.persist(objects_merge=to_add)

    pool.close()
    pool.join()

if __name__ == '__main__':
    logging.getLogger().setLevel(level=logging.INFO)

    # model="gpt-4-0125-preview"
    model_name = LLMName.OPENAI_GPT_35_TURBO
    perform_extraction(model_name=model_name, max_count=25000, pool_size=50 )
    logging.info(f"{len(errors)} errors")
    if len(errors) > 0:
        with open("errors.txt", "w") as file:
            file.write(json.dumps(errors))
    logging.info("Fact extraction done, exiting...")


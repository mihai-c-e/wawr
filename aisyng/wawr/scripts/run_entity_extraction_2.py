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
from aisyng.wawr.models.graph import Entity, WAWRGraphElementTypes
from aisyng.wawr.scripts._entity_extraction_prompts import entity_extraction_prompt_2
from aisyng.base.llms.base import LLMName



sql_lock = threading.Lock()
openai_api_lock = threading.Lock()
openai_api_usage = {"id": str(uuid4())}
wawr_context: WAWRContext = WAWRContext.create_default()
errors = list()



def _get_model_response(fact_node: GraphNode, abstract_node: GraphNode, model_name: LLMName, max_attempts: int = 5) -> List[Entity]:
    logging.info(f"Extracting entities for node {fact_node.id}/{fact_node.text}")
    template = Template(entity_extraction_prompt_2)
    query = template.render(abstract = abstract_node.text, text=fact_node.text)
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
                logging.info(f"API usage: {openai_api_usage}")

            response = read_json(response)
            extracted_objects = [Entity.model_validate({"name": r}) for r in response]
            break
        except Exception as ex:
            logging.exception(ex)
            attempt += 1
            if attempt > max_attempts:
                raise
            logging.error(f"Attempt {attempt}")
    logging.info(f"Model responded with {len(extracted_objects)} entities")
    return extracted_objects

def create_entity_id(entity: Entity) -> str:
    return f"entity:{entity.name}"

def _extracted_object_to_nodes_and_relationship(
        fact_node: GraphNode,
        entity_obj: Entity,
        index: int,
        nodes: Dict[Tuple, GraphNode],
        relationships: List[GraphRelationship]
) -> None:
    # Add the fact node
    entity_node = GraphNode(
        id=create_entity_id(entity=entity_obj),
        text=entity_obj.name,
        meta=entity_obj,
        type_id=WAWRGraphElementTypes.Entity,
        text_type="entity",
        status=''
    )
    nodes[("", entity_node.id)] = entity_node
    # Add source relationship between fact node and paper abstract
    relationships.append(GraphRelationship(
        id=f"{entity_node.id}-{fact_node.id}",
        from_node=entity_node,
        to_node=fact_node,
        text=WAWRGraphElementTypes.IsExtractedFrom,
        type_id=WAWRGraphElementTypes.IsExtractedFrom
    ))
    # Add the fact type node
    """entity_type_node = GraphNode(
        id=f"entity_type:{entity_obj.type}",
        text=entity_obj.type,
        type_id=GraphElementTypes.FactType
    )
    nodes[("", entity_type_node.id)] = entity_type_node
    # Add the is_a relationship between fact type node and fact node
    relationships.append(GraphRelationship(
        from_node=entity_node,
        to_node=entity_type_node,
        text=GraphElementTypes.IsA,
        type_id=GraphElementTypes.IsA
    ))"""

def _extract_from_one(fact_node: GraphNode, model_name: LLMName) -> List[GraphElement]:
    global errors
    logging.info(f"Extracting entities from {fact_node.id}")
    try:
        global openai_api_lock
        global sql_lock
        global openai_api_usage
        abstract_node = wawr_context.get_persistence().get_graph_element_by_id(fact_node.source_id)
        extracted_objects = _get_model_response(fact_node=fact_node, abstract_node=abstract_node, max_attempts=5, model_name=model_name)
        nodes = dict()
        relationships = list()

        for i, entity_obj in enumerate(extracted_objects):
            _extracted_object_to_nodes_and_relationship(
                fact_node=fact_node,
                entity_obj=entity_obj,
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
        error_info = {'fact': fact_node.id, 'message': traceback.format_exception_only()}
        errors.append(error_info)
        raise

def perform_extraction( model_name: LLMName, max_count: int = 20, pool_size: int = 1, batch_size: int=100):
    facts = wawr_context.get_persistence().get_parents_without_extracted_children(
        parent_type_id=WAWRGraphElementTypes.Fact, limit = max_count
    )
    extract_from_one_partial = partial(_extract_from_one, model_name=model_name)
    pool = ThreadPool(pool_size)
    for i in range(0, len(facts), batch_size):
        logging.info(f"Batch {i} - {i+batch_size} of {len(facts)}")
        results: List[List[GraphElement]] = pool.map(extract_from_one_partial, facts[i:i+batch_size])
        to_add = [ge for ge_list in results for ge in ge_list]
        to_add_uniques = list({ge.id: ge for ge_list in results for ge in ge_list}.values())
        logging.info(f"Received {len(to_add)} records, {len(to_add_uniques)} uniques, saving to database")
        with sql_lock:
            wawr_context.get_persistence().sqli.persist(objects_merge=to_add_uniques)

    pool.close()
    pool.join()

if __name__ == '__main__':
    logging.getLogger().setLevel( level=logging.INFO)

    # model="gpt-4-0125-preview"
    # model = "gpt-3.5-turbo-16k"
    model_name = LLMName.OPENAI_GPT_35_TURBO # = "gpt-3.5-turbo"
    # model = "mistral-large-latest"
    perform_extraction(model_name=model_name, max_count=10000000, pool_size=50, batch_size=100000)
    logging.info(f"{len(errors)} errors")
    if len(errors) > 0:
        with open("errors.txt", "w") as file:
            file.write(json.dumps(errors))
    logging.info("Entity extraction done, exiting...")


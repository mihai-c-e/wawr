import concurrent.futures
import logging

from aisyng.base.models.graph import GraphNode
from aisyng.base.models.payload import TopicSolverCallback, TopicSolverBase
from aisyng.wawr.models.models_factory import create_topic_solver_node, create_topic_solver_relationship
from aisyng.wawr.context import WAWRContext

from multiprocessing.pool import Pool

class TopicSolverPersistenceCallback(TopicSolverCallback):
    context: WAWRContext
    topic_solver_node: GraphNode

    def __init__(self, topic_solver_node: GraphNode, context: WAWRContext):
        self.context = context
        self.topic_solver_node = topic_solver_node

    def state_changed(self, topic_solver: TopicSolverBase) -> None:
        if topic_solver.progress == 1.0:
            self.topic_solver_node.text = topic_solver.answer
        self.context.get_persistence().persist(objects_merge=[self.topic_solver_node])


def init_topic_solving(
        topic_node: GraphNode,
        context: WAWRContext,
        topic_solver: TopicSolverBase = None,
        topic_solver_node: GraphNode = None
) -> GraphNode:
    if (topic_solver is None) == (topic_solver_node is None):
        raise ValueError("Provide either topic_solver_node or topic_solver")
    if topic_solver_node is None:
        topic_solver_node = create_topic_solver_node(topic_node=topic_node, topic_solver=topic_solver)

    relationship = create_topic_solver_relationship(topic_solver_node=topic_solver_node, topic_node=topic_node)
    context.get_persistence().persist(objects_merge=[topic_node, topic_solver_node, relationship])
    return topic_solver_node

def solve_topic(
        topic_node: GraphNode,
        context: WAWRContext,
        topic_solver_node: GraphNode = None,
) -> GraphNode:
    callback = TopicSolverPersistenceCallback(topic_solver_node=topic_solver_node, context=context)
    topic_solver = topic_solver_node.meta
    topic_solver.add_callback(callback)
    topic_solver.solve(ask=topic_node.text, ask_embedding=None, context=context)
    return topic_solver_node

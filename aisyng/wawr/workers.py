from aisyng.base.models import GraphNode
from aisyng.wawr.models.topic import TopicSolverBase, TopicSolverCallback
from aisyng.wawr.models.models_factory import create_topic_solver_node, create_topic_solver_relationship
from aisyng.wawr.context import WAWRContext

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

def solve_topic(topic_node: GraphNode, topic_solver: TopicSolverBase, context: WAWRContext) -> GraphNode:
    topic_solver_node = create_topic_solver_node(topic_node=topic_node, topic_solver=topic_solver)
    relationship = create_topic_solver_relationship(topic_solver_node=topic_solver_node, topic_node=topic_node)
    context.get_persistence().persist(objects_merge=[topic_node, topic_solver_node, relationship])

    callback = TopicSolverPersistenceCallback(topic_solver_node=topic_solver_node, context=context)
    topic_solver.add_callback(callback)
    topic_solver.solve(ask=topic_node.text, ask_embeddings=None, context=context)
    return topic_solver_node
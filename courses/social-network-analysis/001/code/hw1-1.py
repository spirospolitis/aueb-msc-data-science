import sys
import unittest

from SnaHomework1.Part1 import GraphGenerator
from SnaHomework1.Part1 import GraphEvaluator

# Increase recursion limit.
sys.setrecursionlimit(10000)

graph_generator = GraphGenerator.GraphGenerator()
graph_evaluator = GraphEvaluator.GraphEvaluator()

def has_euler_path(graph):
    vertices = set()

    # FILL HERE
    result, vertices = graph_evaluator.has_euler_path(graph)
    # END FILL HERE

    return result, vertices


def has_euler_circuit(graph):
    # FILL HERE
    result = graph_evaluator.has_euler_circuit(graph)
    # END FILL HERE

    return result


class TestEulerMethods(unittest.TestCase):
    NUM_NODES = None

    def test_has_euler_path_but_not_circuit(self):
        # FILL HERE
        graph = graph_generator.generate_has_euler_path_but_not_circuit(num_nodes=self.NUM_NODES)
        # END FILL HERE

        result, vertices = has_euler_path(graph)

        self.assertTrue(result)
        self.assertEqual(len(vertices), 2)

    def test_does_not_have_euler_path(self):
        # FILL HERE
        graph = graph_generator.generate_does_not_have_euler_path(num_nodes=self.NUM_NODES)
        # END FILL HERE

        result, vertices = has_euler_path(graph)

        self.assertFalse(result)
        self.assertEqual(len(vertices), 0)

    def test_has_euler_circuit(self):
        # FILL HERE
        graph = graph_generator.generate_has_euler_circuit(num_nodes=self.NUM_NODES)
        # END FILL HERE

        result = has_euler_circuit(graph)

        self.assertTrue(result)
        self.assertTrue(graph.GetNodes()>=1000)

    def test_does_not_have_euler_circuit(self):
        # FILL HERE
        graph = graph_generator.generate_does_not_have_euler_circuit(num_nodes=self.NUM_NODES)
        # END FILL HERE

        result = has_euler_circuit(graph)

        self.assertFalse(result)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestEulerMethods.NUM_NODES = int(sys.argv.pop())
    unittest.main()

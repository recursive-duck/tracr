# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for compiler.craft_graph_to_model."""

from absl.testing import absltest
from absl.testing import parameterized
import networkx as nx
from tracr.compiler import craft_graph_to_model
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.craft import bases
from tracr.craft.chamber import categorical_attn
from tracr.craft.chamber import categorical_mlp
from tracr.rasp import rasp


class CraftAllocateModulesToLayersTest(parameterized.TestCase):

  def _get_dummy_block(self, block_type):
    if block_type == "ATTN":
      return categorical_attn.categorical_attn(
          query_space=bases.VectorSpaceWithBasis.from_names(["query"]),
          key_space=bases.VectorSpaceWithBasis.from_names(["bos", "key"]),
          value_space=bases.VectorSpaceWithBasis.from_names(["bos", "value"]),
          output_space=bases.VectorSpaceWithBasis.from_names(["output"]),
          bos_space=bases.VectorSpaceWithBasis.from_names(["bos"]),
          one_space=bases.VectorSpaceWithBasis.from_names(["one"]),
          attn_fn=lambda x, y: True,
      )
    elif block_type == "MLP":
      return categorical_mlp.map_categorical_mlp(
          input_space=bases.VectorSpaceWithBasis.from_names(["input"]),
          output_space=bases.VectorSpaceWithBasis.from_names(["output"]),
          operation=lambda x: x,
      )
    else:
      return None

  def test_compute_computational_depth_returns_expected_result(self):
    """Creates a graph and checks the longest path for each node."""

    # Node IDs:
    # 0 -- 1 -- 2 -- 3 ------------  4
    #               /              /
    # 5 -- 6 ---------- 7 -- 8 -- 9
    #
    # 10
    # Expected return values:
    # 0 -- 1 -- 2 -- 3 ------------  5
    #               /              /
    # 0 -- 1 ---------- 2 -- 3 -- 4
    #
    # -1

    graph = nx.DiGraph()
    node_ids = list(range(11))
    expected_results = [0, 1, 2, 3, 5, 0, 1, 2, 3, 4, -1]
    for node_id, res in zip(node_ids, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              "expected_result": res
          })
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(5, 6)
    graph.add_edge(6, 7)
    graph.add_edge(7, 8)
    graph.add_edge(8, 9)
    graph.add_edge(6, 3)
    graph.add_edge(9, 4)
    sources = [graph.nodes[0], graph.nodes[5]]

    computational_depth = craft_graph_to_model.compute_computational_depth(
        graph, [src[nodes.ID] for src in sources]
    )
    for node_id, node in graph.nodes.items():
      self.assertEqual(computational_depth[node_id], node["expected_result"])

  def test_allocate_modules_to_layers_returns_expected_result(self):
    """Creates a graph and checks if the correct layer assignment is returned."""

    # Computation Graph:
    # INPUT -- ATTN -- MLP -- ATTN ------ MLP -- OUTPUT
    #           /           /          /
    # INPUT -- MLP --- MLP          ATTN
    #                      \      /
    #                        ATTN
    # Node IDs:
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5
    #         /     /     /
    # 6 -- 7 ---- 8      9
    #               \   /
    #                10
    # Expected layer allocation:
    # -1 -- 0 -- 3 -- 4 -- 7 -- -1
    #         /     /     /
    # -1 -- 1 --- 3      6
    #               \   /
    #                 4

    graph = nx.DiGraph()
    node_ids = list(range(11))
    types = [
        "INPUT", "ATTN", "MLP", "ATTN", "MLP", "OUTPUT", "INPUT", "MLP", "MLP",
        "ATTN", "ATTN"
    ]
    expected_results = [-1, 0, 3, 4, 7, -1, -1, 1, 3, 6, 4]
    for node_id, node_type, res in zip(node_ids, types, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              nodes.MODEL_BLOCK: self._get_dummy_block(node_type),
              "expected_result": res
          })

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(6, 7)
    graph.add_edge(7, 2)
    graph.add_edge(7, 8)
    graph.add_edge(8, 3)
    graph.add_edge(8, 10)
    graph.add_edge(9, 4)
    graph.add_edge(10, 9)

    craft_graph = rasp_to_graph.ExtractRaspGraphOutput(
        graph=graph,
        sink=graph.nodes[10],
        sources=[graph.nodes[0], graph.nodes[6]])

    layer_allocation = craft_graph_to_model._allocate_modules_to_layers(
        craft_graph.graph, craft_graph.sources)
    for node_id, node in graph.nodes.items():
      self.assertEqual(layer_allocation[node_id], node["expected_result"])

  def test_allocate_modules_to_layers_returns_expected_result_for_chain(self):
    """Tests a chain of alternating attention layers and MLPs."""

    # Computation Graph:
    # INPUT -- ATTN -- MLP -- ATTN -- MLP -- OUTPUT
    # Node IDs:
    # 0 -- 1 -- 2 -- 3 -- 4 -- 5
    # Expected layer allocation:
    # -1 -- 0 -- 1 -- 2 -- 3 -- -1

    graph = nx.DiGraph()
    node_ids = list(range(11))
    types = ["INPUT", "ATTN", "MLP", "ATTN", "MLP", "OUTPUT"]
    expected_results = [-1, 0, 1, 2, 3, -1]
    for node_id, node_type, res in zip(node_ids, types, expected_results):
      graph.add_node(
          node_id, **{
              nodes.ID: node_id,
              nodes.EXPR: rasp.ConstantSOp(1),
              nodes.MODEL_BLOCK: self._get_dummy_block(node_type),
              "expected_result": res
          })

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)

    craft_graph = rasp_to_graph.ExtractRaspGraphOutput(
        graph=graph, sink=graph.nodes[5], sources=[graph.nodes[0]])

    layer_allocation = craft_graph_to_model._allocate_modules_to_layers(
        craft_graph.graph, craft_graph.sources)
    for node_id, node in graph.nodes.items():
      self.assertEqual(layer_allocation[node_id], node["expected_result"])


class _SuccessorBudgetDiGraph(nx.DiGraph):
  """A DiGraph that fails the test if successors() is called too often."""

  def __init__(self, successor_budget):
    super().__init__()
    self.successor_budget = successor_budget
    self.successor_calls = 0

  def successors(self, n):
    self.successor_calls += 1
    if self.successor_calls > self.successor_budget:
      raise AssertionError(
          f"successors() called more than {self.successor_budget} times; "
          "depth computation is enumerating paths instead of visiting nodes."
      )
    return super().successors(n)


class ComputeComputationalDepthTest(absltest.TestCase):

  def _diamond_chain(self, levels):
    """Builds a chain of diamonds with 2**levels paths from node 0 to the end.

    Each level is: prev -> selector -> join, and prev -> sop -> join, where the
    selector is a non-SOp and the sop is an SOp. The last join has depth
    2 * levels. Node `orphan` is disconnected and has depth -1.
    """
    num_nodes = 1 + 3 * levels + 1
    graph = _SuccessorBudgetDiGraph(successor_budget=20 * (num_nodes + 1))
    select = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    graph.add_node(0, **{nodes.ID: 0, nodes.EXPR: rasp.tokens})
    prev = 0
    next_id = 1
    for _ in range(levels):
      selector_id, sop_id, join_id = next_id, next_id + 1, next_id + 2
      next_id += 3
      sop = rasp.Map(lambda t: t, rasp.tokens)
      join = rasp.SequenceMap(lambda a, b: a, sop, rasp.tokens)
      graph.add_node(selector_id, **{nodes.ID: selector_id, nodes.EXPR: select})
      graph.add_node(sop_id, **{nodes.ID: sop_id, nodes.EXPR: sop})
      graph.add_node(join_id, **{nodes.ID: join_id, nodes.EXPR: join})
      graph.add_edge(prev, selector_id)
      graph.add_edge(prev, sop_id)
      graph.add_edge(selector_id, join_id)
      graph.add_edge(sop_id, join_id)
      prev = join_id
    orphan = next_id
    graph.add_node(orphan, **{nodes.ID: orphan, nodes.EXPR: rasp.indices})
    return graph, prev, orphan

  def test_diamond_chain_depth_is_linear_and_correct(self):
    levels = 40
    graph, last_join, orphan = self._diamond_chain(levels)
    # Repeating the source (as the old traversal produced) must not change
    # the result.
    depth = craft_graph_to_model.compute_computational_depth(
        graph, [0] * 8
    )

    self.assertEqual(depth[0], 0)
    self.assertEqual(depth[last_join], 2 * levels)
    self.assertEqual(depth[orphan], -1)

  def test_depth_is_maximum_over_paths(self):
    # 0 -> 1 (SOp) -> 3 (SOp), and 0 -> 2 (non-SOp) -> 3 (SOp).
    # The longer path through the SOp 1 gives depth 2 to node 3, not 1.
    graph = nx.DiGraph()
    select = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    graph.add_node(0, **{nodes.ID: 0, nodes.EXPR: rasp.tokens})
    graph.add_node(1, **{nodes.ID: 1, nodes.EXPR: rasp.tokens})
    graph.add_node(2, **{nodes.ID: 2, nodes.EXPR: select})
    graph.add_node(3, **{nodes.ID: 3, nodes.EXPR: rasp.tokens})
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(1, 3)
    graph.add_edge(2, 3)

    depth = craft_graph_to_model.compute_computational_depth(graph, [0])

    self.assertEqual(depth, {0: 0, 1: 1, 2: 0, 3: 2})


if __name__ == "__main__":
  absltest.main()

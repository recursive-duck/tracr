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
"""Tests for compiler.rasp_to_graph."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.rasp import rasp


class ExtractRaspGraphTest(parameterized.TestCase):

  def test_primitives_have_no_edges(self):
    tokens_graph = rasp_to_graph.extract_rasp_graph(rasp.tokens).graph
    self.assertEmpty(tokens_graph.edges)

    indices_graph = rasp_to_graph.extract_rasp_graph(rasp.indices).graph
    self.assertEmpty(indices_graph.edges)

    full_graph = rasp_to_graph.extract_rasp_graph(rasp.Full(1)).graph
    self.assertEmpty(full_graph.edges)

  def test_one_edge(self):
    program = rasp.Map(lambda x: x + 1, rasp.tokens)

    graph = rasp_to_graph.extract_rasp_graph(program).graph

    self.assertLen(graph.edges, 1)
    (u, v), = graph.edges
    self.assertEqual(graph.nodes[u][nodes.EXPR], rasp.tokens)
    self.assertEqual(graph.nodes[v][nodes.EXPR], program)

  def test_aggregate(self):
    program = rasp.Aggregate(
        rasp.Select(rasp.tokens, rasp.indices, rasp.Comparison.EQ),
        rasp.indices,
    )

    extracted = rasp_to_graph.extract_rasp_graph(program)

    # Expected graph:
    #
    # indices \ --------
    #          \         \
    #           select -- program
    # tokens  /

    self.assertLen(extracted.graph.edges, 4)
    self.assertEqual(extracted.sink[nodes.EXPR], program)
    for source in extracted.sources:
      self.assertIn(
          source[nodes.EXPR],
          [rasp.tokens, rasp.indices],
      )

  def test_sources_are_listed_once_in_first_seen_order(self):
    # `tokens` is reachable along several paths, but must be a source only once.
    program = rasp.SequenceMap(
        lambda x, y: x + y,
        rasp.Map(lambda x: x + 1, rasp.tokens),
        rasp.SequenceMap(
            lambda x, y: x * y,
            rasp.indices,
            rasp.Map(lambda x: x - 1, rasp.tokens),
        ),
    )

    extracted = rasp_to_graph.extract_rasp_graph(program)

    self.assertEqual(
        [source[nodes.EXPR] for source in extracted.sources],
        [rasp.tokens, rasp.indices],
    )

class _BudgetedExpr:
  """Shared counter: fails the test if too many children reads happen."""

  def __init__(self, budget):
    self.budget = budget
    self.reads = 0

  def record_read(self):
    self.reads += 1
    if self.reads > self.budget:
      raise AssertionError(
          f"children were read more than {self.budget} times; traversal is "
          "visiting shared expressions once per path."
      )


class _CountedMap(rasp.Map):
  """A Map whose child reads are counted against a shared budget."""

  def __init__(self, f, inner, counter):
    super().__init__(f, inner, simplify=False)
    self._counter = counter

  @property
  def children(self):
    self._counter.record_read()
    return super().children


class _CountedSequenceMap(rasp.SequenceMap):
  """A SequenceMap whose child reads are counted against a shared budget."""

  def __init__(self, f, fst, snd, counter):
    super().__init__(f, fst, snd)
    self._counter = counter

  @property
  def children(self):
    self._counter.record_read()
    return super().children


class ExtractRaspGraphDiamondTest(absltest.TestCase):

  def test_diamond_chain_visits_each_expression_once(self):
    # Every path through this chain goes through 2**depth paths to `tokens`,
    # so a traversal that revisits shared expressions per path would read
    # children far more than the budget (which is linear in the chain length).
    depth = 60
    counter = _BudgetedExpr(budget=50 * depth)
    x = rasp.Map(lambda t: t % 2, rasp.tokens)
    for _ in range(depth):
      x = _CountedSequenceMap(
          lambda a, b: (a + b) % 2,
          x,
          _CountedMap(lambda t: 1 - t, x, counter),
          counter,
      )

    extracted = rasp_to_graph.extract_rasp_graph(x)

    self.assertLessEqual(counter.reads, 2 * depth + 1)
    self.assertLen(extracted.sources, 1)
    self.assertEqual(extracted.sources[0][nodes.EXPR], rasp.tokens)
    self.assertEqual(extracted.graph.number_of_nodes(), 2 * depth + 2)


if __name__ == "__main__":
  absltest.main()

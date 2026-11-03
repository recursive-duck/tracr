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
"""Converting a RaspExpr to a graph."""

import dataclasses
import queue
from typing import List

import networkx as nx
from tracr.compiler import nodes
from tracr.rasp import rasp

Node = nodes.Node
NodeID = nodes.NodeID


@dataclasses.dataclass
class ExtractRaspGraphOutput:
  graph: nx.DiGraph
  sink: Node  # the program's output.
  sources: List[Node]  # the primitive S-Ops.


def extract_rasp_graph(tip: rasp.SOp) -> ExtractRaspGraphOutput:
  """Converts a RASP program into a graph representation.

  Each expression is visited once, so shared subexpressions do not blow up the
  traversal. Sources are listed once each, in first-seen order.
  """
  expr_queue = queue.Queue()
  graph = nx.DiGraph()
  sources: List[NodeID] = []
  visited_ids = set()

  def ensure_node(expr: rasp.RASPExpr) -> NodeID:
    """Finds or creates a graph node corresponding to expr; returns its ID."""
    node_id = expr.label
    if node_id not in graph:
      graph.add_node(node_id, **{nodes.ID: node_id, nodes.EXPR: expr})

    return node_id

  def enqueue_once(expr: rasp.RASPExpr):
    node_id = expr.label
    if node_id not in visited_ids:
      visited_ids.add(node_id)
      expr_queue.put(expr)

  # Breadth-first search over the RASP expression graph.

  def visit_raspexpr(expr: rasp.RASPExpr):
    parent_id = ensure_node(expr)
    children = expr.children

    for child_expr in children:
      child_id = ensure_node(child_expr)
      graph.add_edge(child_id, parent_id)
      enqueue_once(child_expr)

    if not children:
      sources.append(graph.nodes[parent_id])

  sink = graph.nodes[ensure_node(tip)]
  enqueue_once(tip)
  while not expr_queue.empty():
    visit_raspexpr(expr_queue.get())

  return ExtractRaspGraphOutput(graph=graph, sink=sink, sources=sources)

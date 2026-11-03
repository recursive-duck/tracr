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
"""Inferring the vector spaces taken on by certain operations."""

import dataclasses
import fractions as fractions_lib
import itertools
from typing import Set

import networkx as nx
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft.chamber import binary_mlp
from tracr.rasp import rasp
from tracr.utils import errors

Node = nodes.Node


def _numerical_aggregate_value_set(
    sop_value_set: Set[rasp.Value], max_seq_len: int
) -> Set[rasp.Value]:
  """Returns all values the mean of binary values can take.

  The mean of n values from {0, 1} is k / n for every 0 <= k <= n, not just
  1 / n. Averaging over fewer than max_seq_len positions is possible, so n
  ranges over 1..max_seq_len. Only the distinct (reduced) fractions are kept,
  so that there are no near-duplicates such as 1/3 and 2/6. 0 is always in the
  set because it is the default output of an Aggregate that selects nothing.

  Args:
    sop_value_set: Values of the averaged S-Op, a subset of {0, 1}.
    max_seq_len: Maximum number of averaged positions.
  """
  has_zero = any(x == 0 for x in sop_value_set)
  has_one = any(x == 1 for x in sop_value_set)
  if has_zero and has_one:
    means = {
        fractions_lib.Fraction(k, n)
        for n in range(1, max_seq_len + 1)
        for k in range(n + 1)
    }
  elif has_one:
    # Every selected value is 1, so the mean is always exactly 1.
    means = {fractions_lib.Fraction(1)}
  else:
    means = set()
  means.add(fractions_lib.Fraction(0))
  return {float(m) for m in means}


@dataclasses.dataclass
class InferBasesOutput:
  graph: nx.DiGraph


def infer_bases(
    graph: nx.DiGraph,
    sink: Node,
    vocab: Set[rasp.Value],
    max_seq_len: int,
) -> None:
  """Infers in-place the possible output values and vector bases of the SOps."""

  def compute_value_set(sop: rasp.SOp) -> Set[rasp.Value]:
    """Computes value set using already-computed predecessor value sets."""
    if isinstance(sop, rasp.TokensType):
      return vocab
    elif isinstance(sop, rasp.IndicesType):
      return set(range(max_seq_len))
    elif isinstance(sop, rasp.SelectorWidth):
      return set(range(0, max_seq_len + 1))
    elif isinstance(sop, rasp.Full):
      return {sop.fill}
    elif isinstance(sop, rasp.Map):
      inner_value_set = graph.nodes[sop.inner.label][nodes.VALUE_SET]
      out = set()
      for x in inner_value_set:
        res = errors.ignoring_arithmetic_errors(sop.f)(x)
        if res is not None:
          out.add(res)
      return out
    elif isinstance(sop, rasp.SequenceMap):
      f_ignore_error = errors.ignoring_arithmetic_errors(sop.f)
      fst_value_set = graph.nodes[sop.fst.label][nodes.VALUE_SET]
      snd_value_set = graph.nodes[sop.snd.label][nodes.VALUE_SET]
      out = set()
      for l, r in itertools.product(fst_value_set, snd_value_set):
        res = f_ignore_error(l, r)
        if res is not None:
          out.add(res)
      return out
    elif isinstance(sop, rasp.Aggregate):
      if rasp.is_categorical(sop) or rasp.is_binary(sop):
        # Simply pass on the value set of the underlying S-Op.
        return graph.nodes[sop.sop.label][nodes.VALUE_SET]
      elif rasp.is_numerical(sop):
        # TODO(b/255936408): This doesn't work if we average arbitrary values.
        # But most examples only average binary variables.
        sop_value_set = graph.nodes[sop.sop.label][nodes.VALUE_SET]
        # Compare values, not int(x): int() truncates, so 0.5 or 1.9 would
        # pass as binary and get a value set that misses most of their means.
        if not all(x == 0 or x == 1 for x in sop_value_set):
          raise NotImplementedError(
              "Attention patterns can currently only "
              "average binary variables. Not:", sop_value_set)

        return _numerical_aggregate_value_set(sop_value_set, max_seq_len)
    raise ValueError(f"Unsupported S-Op: {sop}")

  binary_labels = []
  for node_id in nx.dfs_postorder_nodes(graph.reverse(), sink[nodes.ID]):
    expr = graph.nodes[node_id][nodes.EXPR]

    if not isinstance(expr, rasp.SOp):
      # Only S-Ops have output vector spaces.
      continue

    value_set = compute_value_set(expr)
    graph.nodes[node_id][nodes.VALUE_SET] = value_set

    if rasp.is_categorical(expr):
      out_space = bases.VectorSpaceWithBasis.from_values(expr.label, value_set)
    elif rasp.is_numerical(expr):
      out_space = bases.VectorSpaceWithBasis.from_names([expr.label])
    elif rasp.is_binary(expr):
      binary_labels.append(expr.label)
      continue
    else:
      raise ValueError(f"Unsupported S-Op type: {expr.type}")
    graph.nodes[node_id][nodes.OUTPUT_BASIS] = out_space.basis

  if binary_labels:
    _assign_binary_codes(graph, binary_labels)


def _assign_binary_codes(graph: nx.DiGraph, binary_labels) -> None:
  """Gives every binary S-Op a bit code and its bit directions as basis.

  S-Ops whose bits must agree share one code over the union of their value
  sets: the query and key of a Select (equality is a dot product of codes) and
  the input and output of an Aggregate (attention copies the bits).
  """
  parent = {label: label for label in binary_labels}

  def find(label):
    while parent[label] != label:
      parent[label] = parent[parent[label]]
      label = parent[label]
    return label

  def union(a, b):
    if a in parent and b in parent:
      parent[find(a)] = find(b)

  for node_id in graph.nodes:
    expr = graph.nodes[node_id][nodes.EXPR]
    if isinstance(expr, rasp.Select):
      union(expr.queries.label, expr.keys.label)
    elif isinstance(expr, rasp.Aggregate):
      union(expr.label, expr.sop.label)

  group_values = {}
  for label in binary_labels:
    group_values.setdefault(find(label), set()).update(
        graph.nodes[label][nodes.VALUE_SET])
  for label in binary_labels:
    code = binary_mlp.code_from_values(label, group_values[find(label)])
    graph.nodes[label][nodes.BINARY_CODE] = code
    graph.nodes[label][nodes.OUTPUT_BASIS] = code.space.basis

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
"""Tests for compiler.lib_arith.

Each program reads a = tokens and b = indices as numbers 0..2^w-1, so with
max_seq_len 2^w the inputs [(k + j) % 2^w for j in range(2^w)], k = 0..2^w-1,
cover every pair (a, b). The compiled model must match rasp.evaluate on all of
them, and rasp.evaluate must match integer arithmetic.
"""

import math

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import compiling
from tracr.compiler import craft_graph_to_model
from tracr.compiler import lib_arith
from tracr.compiler import nodes
from tracr.compiler import rasp_to_graph
from tracr.rasp import rasp

_BOS = "BOS"
_WIDTHS = (1, 2, 3, 4)
_DEPTH_WIDTH = 16


def _a(w):
  return lib_arith.to_bits(rasp.tokens, w)


def _b(w):
  return lib_arith.to_bits(rasp.indices, w)


# name -> (program builder of width w, integer function of (a, b)).
_OPS = {
    "ripple_kpg": (
        lambda w: lib_arith.from_bits(
            lib_arith.ripple_add(_a(w), _b(w), lib_arith.KPG)),
        lambda a, b: a + b),
    "ripple_pair": (
        lambda w: lib_arith.from_bits(
            lib_arith.ripple_add(_a(w), _b(w), lib_arith.PAIR)),
        lambda a, b: a + b),
    "ripple_bits": (
        lambda w: lib_arith.from_bits(
            lib_arith.ripple_add(_a(w), _b(w), lib_arith.BITS)),
        lambda a, b: a + b),
    "kogge_stone": (
        lambda w: lib_arith.from_bits(lib_arith.kogge_stone_add(_a(w), _b(w))),
        lambda a, b: a + b),
    "increment_ripple": (
        lambda w: lib_arith.from_bits(
            lib_arith.increment(_a(w), lib_arith.RIPPLE)),
        lambda a, b: a + 1),
    "increment_kogge_stone": (
        lambda w: lib_arith.from_bits(
            lib_arith.increment(_a(w), lib_arith.KOGGE_STONE)),
        lambda a, b: a + 1),
    "less_than_ripple": (
        lambda w: lib_arith.less_than(_a(w), _b(w), lib_arith.RIPPLE),
        lambda a, b: int(a < b)),
    "less_than_tree": (
        lambda w: lib_arith.less_than(_a(w), _b(w), lib_arith.KOGGE_STONE),
        lambda a, b: int(a < b)),
    "from_bits": (
        lambda w: lib_arith.from_bits(_a(w)),
        lambda a, b: a),
}


def _all_pair_inputs(w):
  n = 2**w
  return [[(k + j) % n for j in range(n)] for k in range(n)]


def _cases():
  return [dict(testcase_name=f"{name}_w{w}", name=name, w=w)
          for name in _OPS for w in _WIDTHS]


def _sop_depth(program):
  """Longest chain of S-Ops from the inputs to `program` (MLP layers)."""
  extracted = rasp_to_graph.extract_rasp_graph(program)
  depth = craft_graph_to_model.compute_computational_depth(
      extracted.graph, [src[nodes.ID] for src in extracted.sources])
  return depth[extracted.sink[nodes.ID]]


class LibArithTest(parameterized.TestCase):

  @parameterized.named_parameters(*_cases())
  def test_compiled_matches_rasp_and_integers_on_all_pairs(self, name, w):
    build, reference = _OPS[name]
    program = build(w)
    n = 2**w
    model = compiling.compile_rasp_to_model(
        program, vocab=set(range(n)), max_seq_len=n, compiler_bos=_BOS)
    for seq in _all_pair_inputs(w):
      expected = [reference(a, b) for b, a in enumerate(seq)]
      self.assertEqual(rasp.evaluate(program, seq), expected)
      self.assertEqual(model.apply([_BOS] + seq).decoded[1:], expected)

  def test_bits_are_two_valued_and_carries_never_propagate(self):
    w = 4
    a, b = _a(w), _b(w)
    seqs = _all_pair_inputs(w)
    for out in (lib_arith.ripple_add(a, b, lib_arith.KPG),
                lib_arith.kogge_stone_add(a, b)):
      for bit in out:
        values = {v for seq in seqs for v in rasp.evaluate(bit, seq)}
        self.assertLessEqual(values, {0, 1})

  @parameterized.named_parameters(
      dict(testcase_name="ripple_kpg",
           build=lambda a, b: lib_arith.ripple_add(a, b, lib_arith.KPG),
           depth=lambda w: w),
      dict(testcase_name="ripple_pair",
           build=lambda a, b: lib_arith.ripple_add(a, b, lib_arith.PAIR),
           depth=lambda w: w),
      dict(testcase_name="ripple_bits",
           build=lambda a, b: lib_arith.ripple_add(a, b, lib_arith.BITS),
           depth=lambda w: 2 * w - 1),
      dict(testcase_name="kogge_stone",
           build=lib_arith.kogge_stone_add,
           depth=lambda w: math.ceil(math.log2(w)) + 2),
  )
  def test_adder_depth(self, build, depth):
    w = _DEPTH_WIDTH
    out = build(_a(w), _b(w))
    # Subtract the to_bits Maps.
    self.assertEqual(max(_sop_depth(bit) for bit in out) - 1, depth(w))

  def test_less_than_tree_depth_is_logarithmic(self):
    w = _DEPTH_WIDTH
    tree = lib_arith.less_than(_a(w), _b(w), lib_arith.KOGGE_STONE)
    chain = lib_arith.less_than(_a(w), _b(w), lib_arith.RIPPLE)
    self.assertEqual(_sop_depth(tree) - 1, math.ceil(math.log2(w)) + 1)
    self.assertEqual(_sop_depth(chain) - 1, w)

  def test_rejects_bad_arguments(self):
    a = _a(2)
    with self.assertRaises(ValueError):
      lib_arith.ripple_add(a, a[:1])
    with self.assertRaises(ValueError):
      lib_arith.ripple_add(a, a, fusion="nope")
    with self.assertRaises(ValueError):
      lib_arith.increment(a, network="nope")
    with self.assertRaises(ValueError):
      lib_arith.from_bits([])


if __name__ == "__main__":
  absltest.main()

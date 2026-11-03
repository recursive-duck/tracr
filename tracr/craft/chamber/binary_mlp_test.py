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
"""Tests for chamber.binary_mlp."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.craft import bases
from tracr.craft.chamber import binary_mlp

_ONE = bases.VectorSpaceWithBasis.from_names(["one"])


def _embed(space, parts):
  """Residual vector over `space` holding the given (sub-vector) parts."""
  out = space.null_vector()
  for part in parts:
    out = out + part.project(space)
  return out


class CodeTest(parameterized.TestCase):

  @parameterized.parameters((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (256, 8),
                            (257, 9), (4096, 12))
  def test_num_bits(self, num_values, expected):
    self.assertEqual(binary_mlp.num_bits(num_values), expected)

  @parameterized.parameters(2, 3, 7, 16, 33)
  def test_codes_are_distinct_and_only_match_themselves(self, num_values):
    n_bits = binary_mlp.num_bits(num_values)
    codes = binary_mlp.code_matrix(num_values, n_bits)
    gram = codes @ codes.T
    np.testing.assert_array_equal(np.diag(gram), n_bits)
    off = gram[~np.eye(num_values, dtype=bool)]
    if off.size:
      self.assertLessEqual(off.max(), n_bits - 2)

  def test_code_from_values_uses_sorted_order(self):
    code = binary_mlp.code_from_values("x", {3, 1, 2})
    self.assertEqual(code.values, (1, 2, 3))
    np.testing.assert_array_equal(code.code(1), [-1, -1])
    np.testing.assert_array_equal(code.code(2), [1, -1])
    np.testing.assert_array_equal(code.code(3), [-1, 1])


class LookupMlpTest(parameterized.TestCase):

  @parameterized.parameters("binary", "categorical")
  def test_map_reads_and_writes_codes_exactly(self, input_kind):
    in_values = list(range(6))
    f = lambda x: (3 * x) % 5
    in_code = binary_mlp.code_from_values("x", in_values)
    in_space = bases.VectorSpaceWithBasis.from_values("x", in_values)
    out_code = binary_mlp.code_from_values("y", {f(x) for x in in_values})
    if input_kind == "binary":
      in_op = binary_mlp.BinaryOperand(in_code)
    else:
      in_op = binary_mlp.CategoricalOperand(in_space)
    mlp = binary_mlp.lookup_mlp([in_op], binary_mlp.BinaryOperand(out_code), f,
                                _ONE, "hidden")
    for x in in_values:
      if input_kind == "binary":
        x_vec = in_code.vector(x)
      else:
        x_vec = in_space.vector_from_basis_direction(
            bases.BasisDirection("x", x))
      res = _embed(mlp.residual_space,
                   [x_vec, _ONE.vector_from_basis_direction(_ONE.basis[0])])
      out = mlp.apply(res).project(out_code.space)
      np.testing.assert_array_equal(out.magnitudes, out_code.code(f(x)))

  def test_zero_input_writes_nothing(self):
    code = binary_mlp.code_from_values("x", range(5))
    mlp = binary_mlp.lookup_mlp([binary_mlp.BinaryOperand(code)],
                                binary_mlp.BinaryOperand(code), lambda x: x,
                                _ONE, "hidden")
    res = _embed(mlp.residual_space,
                 [_ONE.vector_from_basis_direction(_ONE.basis[0])])
    np.testing.assert_array_equal(mlp.apply(res).magnitudes, 0)

  def test_sequence_map_of_two_binary_inputs(self):
    xs, ys = range(3), range(5)
    f = lambda x, y: x * y
    x_code = binary_mlp.code_from_values("x", xs)
    y_code = binary_mlp.code_from_values("y", ys)
    out_space = bases.VectorSpaceWithBasis.from_values(
        "z", {f(x, y) for x, y in itertools.product(xs, ys)})
    mlp = binary_mlp.lookup_mlp(
        [binary_mlp.BinaryOperand(x_code), binary_mlp.BinaryOperand(y_code)],
        binary_mlp.CategoricalOperand(out_space), f, _ONE, "hidden")
    for x, y in itertools.product(xs, ys):
      res = _embed(mlp.residual_space, [
          x_code.vector(x), y_code.vector(y),
          _ONE.vector_from_basis_direction(_ONE.basis[0])
      ])
      out = mlp.apply(res).project(out_space)
      expected = out_space.vector_from_basis_direction(
          bases.BasisDirection("z", f(x, y)))
      np.testing.assert_array_equal(out.magnitudes, expected.magnitudes)

  def test_fold_onehot_output_writes_codes(self):
    values = [0, 1, 2, 3, 4]
    onehot = bases.VectorSpaceWithBasis.from_values("y", values)
    code = binary_mlp.code_from_values("y", values)
    in_space = bases.VectorSpaceWithBasis.from_values("x", values)
    mlp = binary_mlp.lookup_mlp([binary_mlp.CategoricalOperand(in_space)],
                                binary_mlp.CategoricalOperand(onehot),
                                lambda x: 4 - x, _ONE, "hidden")
    folded = binary_mlp.fold_onehot_output(mlp, onehot, code)
    for x in values:
      res = _embed(folded.residual_space, [
          in_space.vector_from_basis_direction(bases.BasisDirection("x", x))
      ])
      out = folded.apply(res).project(code.space)
      np.testing.assert_array_equal(out.magnitudes, code.code(4 - x))


if __name__ == "__main__":
  absltest.main()

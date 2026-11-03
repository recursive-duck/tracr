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
"""Tests for chamber.numerical_mlp."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.craft import bases
from tracr.craft import tests_common
from tracr.craft.chamber import numerical_mlp
from tracr.utils import errors


class NumericalMlpTest(tests_common.VectorFnTestCase):

  @parameterized.parameters([
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=2,
          function=lambda x: x,
          result=2),
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=2,
          function=lambda x: x**2,
          result=4),
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=2,
          function=lambda x: x**3,
          result=8),
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=-2,
          function=lambda x: x,
          result=-2),
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=-2,
          function=lambda x: x**2,
          result=4),
      dict(
          in_value_set={-2, -2, -1, 0, 1, 2, 3},
          x=-2,
          function=lambda x: x**3,
          result=-8),
  ])
  def test_map_numerical_mlp_produces_expected_outcome(self, in_value_set, x,
                                                       function, result):

    input_dir = bases.BasisDirection("input")
    output_dir = bases.BasisDirection("output")
    one_dir = bases.BasisDirection("one")
    input_space = bases.VectorSpaceWithBasis([input_dir])
    output_space = bases.VectorSpaceWithBasis([output_dir])
    one_space = bases.VectorSpaceWithBasis([one_dir])

    mlp = numerical_mlp.map_numerical_mlp(
        f=function,
        input_space=input_space,
        output_space=output_space,
        one_space=one_space,
        input_value_set=in_value_set,
    )

    test_inputs = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir, one_dir],
        magnitudes=np.array([x, 0, 1]))

    expected_results = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir, one_dir],
        magnitudes=np.array([0, result, 0]))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)

  @parameterized.parameters(5, 10, 20)
  def test_map_numerical_mlp_resolves_closely_spaced_values(self, max_len):
    # Values are all means k / n of up to max_len binary values; the smallest
    # gap between them is far below 2 / default large_number.
    in_value_set = {
        k / n for n in range(1, max_len + 1) for k in range(n + 1)
    }
    function = lambda x: x * 10

    input_dir = bases.BasisDirection("input")
    output_dir = bases.BasisDirection("output")
    one_dir = bases.BasisDirection("one")
    input_space = bases.VectorSpaceWithBasis([input_dir])
    output_space = bases.VectorSpaceWithBasis([output_dir])
    one_space = bases.VectorSpaceWithBasis([one_dir])

    mlp = numerical_mlp.map_numerical_mlp(
        f=function,
        input_space=input_space,
        output_space=output_space,
        one_space=one_space,
        input_value_set=in_value_set,
    )

    for x in sorted(in_value_set):
      test_inputs = bases.VectorInBasis(
          basis_directions=[input_dir, output_dir, one_dir],
          magnitudes=np.array([x, 0, 1], dtype=np.float32),
      )
      expected_results = bases.VectorInBasis(
          basis_directions=[input_dir, output_dir, one_dir],
          magnitudes=np.array([0, function(x), 0], dtype=np.float32),
      )
      with self.subTest(x=x):
        self.assertVectorAllClose(mlp.apply(test_inputs), expected_results)

  def test_large_number_is_never_lowered(self):
    # Well separated values: the requested large_number must be kept.
    layer = numerical_mlp._get_discretising_layer(
        input_value_set={0, 1, 2},
        f=lambda x: x,
        hidden_name="h",
        one_direction=bases.BasisDirection("one"),
        large_number=1000,
    )
    weight = layer.action(bases.BasisDirection("input"))
    self.assertEqual(max(weight.magnitudes), 1000)

  def test_effective_large_number_warns_beyond_float32_accuracy(self):
    with self.assertLogs(level="WARNING"):
      numerical_mlp._effective_large_number([0, 1e-6, 1], 100)

  def test_near_duplicates_are_merged_onto_the_simple_fraction(self):
    values = sorted([-5.5e-17, 0.0, 5.5e-17, 1 / 3, 1 / 3 + 1e-16, 1.0])
    self.assertEqual(
        numerical_mlp._merge_near_duplicates(values), [0.0, 1 / 3, 1.0]
    )

  @parameterized.parameters(
      dict(values=[0, 0.001, 1], expected=4 / 0.001),
      dict(values=[0, 1], expected=100),
      dict(values=[3], expected=100),
  )
  def test_effective_large_number(self, values, expected):
    self.assertAlmostEqual(
        numerical_mlp._effective_large_number(values, 100), expected, places=6
    )

  @parameterized.parameters([
      dict(in_value_set={0, 1, 2, 3}, x=1, function=lambda x: 1 / x, result=1),
      dict(
          in_value_set={0, 1, 2, 3}, x=2, function=lambda x: 1 / x, result=0.5),
      dict(
          in_value_set={0, 1, 2, 3},
          x=3,
          function=lambda x: 1 / x,
          result=1 / 3),
  ])
  def test_map_numerical_mlp_logs_warning_and_produces_expected_outcome(
      self, in_value_set, x, function, result):

    input_dir = bases.BasisDirection("input")
    output_dir = bases.BasisDirection("output")
    one_dir = bases.BasisDirection("one")
    input_space = bases.VectorSpaceWithBasis([input_dir])
    output_space = bases.VectorSpaceWithBasis([output_dir])
    one_space = bases.VectorSpaceWithBasis([one_dir])

    with self.assertLogs(level="WARNING"):
      mlp = numerical_mlp.map_numerical_mlp(
          f=function,
          input_space=input_space,
          output_space=output_space,
          one_space=one_space,
          input_value_set=in_value_set,
      )

    test_inputs = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir, one_dir],
        magnitudes=np.array([x, 0, 1]))

    expected_results = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir, one_dir],
        magnitudes=np.array([0, result, 0]))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)

  @parameterized.parameters([
      dict(in_value_set={0, 1, 2, 3}, x=1, function=lambda x: 1 / x, result=1),
      dict(
          in_value_set={0, 1, 2, 3}, x=2, function=lambda x: 1 / x, result=0.5),
      dict(
          in_value_set={0, 1, 2, 3},
          x=3,
          function=lambda x: 1 / x,
          result=1 / 3),
  ])
  def test_map_numerical_to_categorical_mlp_logs_warning_and_produces_expected_outcome(
      self, in_value_set, x, function, result):

    f_ign = errors.ignoring_arithmetic_errors(function)
    out_value_set = {f_ign(x) for x in in_value_set if f_ign(x) is not None}

    in_space = bases.VectorSpaceWithBasis.from_names(["input"])
    out_space = bases.VectorSpaceWithBasis.from_values("output", out_value_set)
    one_space = bases.VectorSpaceWithBasis.from_names(["one"])

    residual_space = bases.join_vector_spaces(in_space, one_space, out_space)
    in_vec = residual_space.vector_from_basis_direction(in_space.basis[0])
    one_vec = residual_space.vector_from_basis_direction(one_space.basis[0])

    with self.assertLogs(level="WARNING"):
      mlp = numerical_mlp.map_numerical_to_categorical_mlp(
          f=function,
          input_space=in_space,
          output_space=out_space,
          input_value_set=in_value_set,
          one_space=one_space,
      )

    test_inputs = x * in_vec + one_vec
    expected_results = out_space.vector_from_basis_direction(
        bases.BasisDirection("output", result))
    test_outputs = mlp.apply(test_inputs).project(out_space)
    self.assertVectorAllClose(test_outputs, expected_results)

  @parameterized.parameters([
      dict(x_factor=1, y_factor=2, x=1, y=1, result=3),
      dict(x_factor=1, y_factor=2, x=1, y=-1, result=-1),
      dict(x_factor=1, y_factor=-1, x=1, y=1, result=0),
      dict(x_factor=1, y_factor=1, x=3, y=5, result=8),
      dict(x_factor=-2, y_factor=-0.5, x=4, y=1, result=-8.5),
  ])
  def test_linear_sequence_map_produces_expected_result(self, x_factor,
                                                        y_factor, x, y, result):

    input1_dir = bases.BasisDirection("input1")
    input2_dir = bases.BasisDirection("input2")
    output_dir = bases.BasisDirection("output")

    mlp = numerical_mlp.linear_sequence_map_numerical_mlp(
        input1_basis_direction=input1_dir,
        input2_basis_direction=input2_dir,
        output_basis_direction=output_dir,
        input1_factor=x_factor,
        input2_factor=y_factor)

    test_inputs = bases.VectorInBasis(
        basis_directions=[input1_dir, input2_dir, output_dir],
        magnitudes=np.array([x, y, 0]))

    expected_results = bases.VectorInBasis(
        basis_directions=[input1_dir, input2_dir, output_dir],
        magnitudes=np.array([0, 0, result]))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)

  @parameterized.parameters([
      dict(x_factor=1, y_factor=2, x=1, result=3),
      dict(x_factor=1, y_factor=-1, x=1, result=0),
  ])
  def test_linear_sequence_map_produces_expected_result_with_same_inputs(
      self, x_factor, y_factor, x, result):

    input_dir = bases.BasisDirection("input")
    output_dir = bases.BasisDirection("output")

    mlp = numerical_mlp.linear_sequence_map_numerical_mlp(
        input1_basis_direction=input_dir,
        input2_basis_direction=input_dir,
        output_basis_direction=output_dir,
        input1_factor=x_factor,
        input2_factor=y_factor)

    test_inputs = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir], magnitudes=np.array([x, 0]))

    expected_results = bases.VectorInBasis(
        basis_directions=[input_dir, output_dir],
        magnitudes=np.array([0, result]))

    test_outputs = mlp.apply(test_inputs)

    self.assertVectorAllClose(test_outputs, expected_results)


if __name__ == "__main__":
  absltest.main()

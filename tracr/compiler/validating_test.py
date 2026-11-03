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
"""Tests for compiler.compilable_evaluator."""

from absl.testing import absltest
from absl.testing import parameterized
from tracr.compiler import compiling
from tracr.compiler import test_cases
from tracr.compiler import validating
from tracr.rasp import rasp


def _unsimplified_and_selector():
  return rasp.selector_and(
      rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
      rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
      simplify=False,
  )


def _unsimplified_not_selector():
  return rasp.selector_not(
      rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
      simplify=False,
  )


# Programs that compile_rasp_to_model used to reject with a raw TypeError,
# AttributeError or ValueError, or a craft-stage NotImplementedError. They must
# now be reported statically with a NotImplementedError.
_UNCOMPILABLE_PROGRAMS = [
    dict(
        testcase_name="full_sop",
        program=rasp.Map(lambda x: x, rasp.Full(1)),
    ),
    dict(
        testcase_name="length_sop",
        program=rasp.Map(lambda x: x, rasp.length),
    ),
    dict(
        testcase_name="aggregate_with_unsimplified_and_selector",
        program=rasp.Aggregate(_unsimplified_and_selector(), rasp.tokens),
    ),
    dict(
        testcase_name="selector_width_with_unsimplified_and_selector",
        program=rasp.SelectorWidth(_unsimplified_and_selector()),
    ),
    dict(
        testcase_name="aggregate_with_or_selector",
        program=rasp.Aggregate(
            rasp.selector_or(
                rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
                rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
                simplify=False,
            ),
            rasp.tokens,
        ),
    ),
    dict(
        testcase_name="selector_width_with_not_selector",
        program=rasp.SelectorWidth(_unsimplified_not_selector()),
    ),
    dict(
        testcase_name="aggregate_with_constant_selector",
        program=rasp.Aggregate(
            rasp.ConstantSelector([[True, True], [False, False]]),
            rasp.tokens,
        ),
    ),
    dict(
        testcase_name="constant_sop",
        program=rasp.ConstantSOp([1, 2, 1]),
    ),
]


class ValidationEvaluatorTest(parameterized.TestCase):

  @parameterized.named_parameters(test_cases.TEST_CASES)
  def test_supported_programs_pass_validation(
      self,
      program,
      test_input,
      **kwargs,
  ):
    del kwargs
    validation_result = validating.validate(program, test_input)
    self.assertEmpty(validation_result)

  @parameterized.named_parameters(test_cases.UNSUPPORTED_TEST_CASES)
  def test_unsupported_programs_fail_validation(
      self,
      program,
      vocab,
      **kwargs,
  ):
    del kwargs
    test_input = sorted(list(vocab))
    validation_result = validating.validate(program, test_input)
    self.assertNotEmpty(validation_result)

  @parameterized.named_parameters(
      dict(
          testcase_name="mean",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE),
              rasp.tokens,
          ),
          test_input=[1, 2, 3, 4],
      ),
      dict(
          testcase_name="prev_mean",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LEQ),
              rasp.tokens,
          ),
          test_input=[1, 2, 3, 4],
      ),
  )
  def test_dynamic_failure_cases_fail_validation(
      self,
      program,
      test_input,
  ):
    # Dynamic test cases are not in the general test case suite because they are
    # not caught at compile time.
    validation_result = validating.validate(program, test_input)
    self.assertNotEmpty(validation_result)

  @parameterized.named_parameters(_UNCOMPILABLE_PROGRAMS)
  def test_uncompilable_programs_fail_static_validation(self, program):
    self.assertNotEmpty(validating.static_validate(program))

  @parameterized.named_parameters(_UNCOMPILABLE_PROGRAMS)
  def test_uncompilable_programs_fail_compilation_early(self, program):
    with self.assertRaisesRegex(
        NotImplementedError, "Unsupported RASP expressions"
    ):
      compiling.compile_rasp_to_model(program, vocab={1, 2}, max_seq_len=3)

  def test_shared_default_names_do_not_hide_child_issues(self):
    # Every Aggregate is named "aggregate", so name-based deduplication used to
    # skip the child `bad` when walking `outer`.
    selector = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ)
    bad = rasp.Aggregate(selector, rasp.tokens, default=5)
    outer = rasp.Aggregate(selector, bad)

    issues = validating.static_validate(outer)

    self.assertLen(issues, 1)
    self.assertIs(issues[0].expr, bad)

  def test_shared_subgraphs_are_walked_once(self):
    # Each level reaches the previous one along two paths, so the graph has
    # 2**64 paths but only 129 distinct expressions. The walk must stay linear.
    expr = rasp.ConstantSOp([1, 2, 1])
    for _ in range(64):
      expr = rasp.SequenceMap(
          lambda x, y: x, expr, rasp.Map(lambda x: x, expr)
      )

    issues = validating.static_validate(expr)

    self.assertLen(issues, 1)
    self.assertIsInstance(issues[0].expr, rasp.ConstantSOp)

  def test_numerical_indices_reason_names_indices(self):
    issues = validating.static_validate(rasp.numerical(rasp.indices))

    self.assertLen(issues, 1)
    self.assertEqual(issues[0].reason, "indices should always be categorical.")


if __name__ == "__main__":
  absltest.main()

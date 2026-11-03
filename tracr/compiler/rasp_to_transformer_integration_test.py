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
"""Integration tests for the full RASP -> transformer compilation."""

import itertools
import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.compiler import test_cases
from tracr.craft import tests_common
from tracr.rasp import rasp

_COMPILER_BOS = "rasp_to_transformer_integration_test_BOS"
_COMPILER_PAD = "rasp_to_transformer_integration_test_PAD"


def _make_fraction_of_ones():
  """Running mean of the binary tokens, i.e. a numerical Aggregate."""
  is_one = rasp.numerical(rasp.Map(lambda t: t == 1, rasp.tokens))
  prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
  return rasp.numerical(rasp.Aggregate(prevs, is_one, default=0))


def _sample_binary_inputs(max_seq_len, num_samples, seed):
  """Seeded sample of binary inputs, half of them of maximum length."""
  rng = random.Random(seed)
  samples = [[1] * max_seq_len, [0, 1] * (max_seq_len // 2)]
  while len(samples) < num_samples:
    length = max_seq_len if rng.random() < 0.5 else rng.randint(1, max_seq_len)
    samples.append([rng.randint(0, 1) for _ in range(length)])
  return samples

# Force float32 precision on TPU, which otherwise defaults to float16.
jax.config.update("jax_default_matmul_precision", "float32")


class CompilerIntegrationTest(tests_common.VectorFnTestCase):

  def assertSequenceEqualWhenExpectedIsNotNone(self, actual_seq, expected_seq):
    for actual, expected in zip(actual_seq, expected_seq):
      if expected is not None and actual != expected:
        self.fail(
            f"{actual_seq} does not match (ignoring Nones) "
            f"expected_seq={expected_seq}"
        )

  @parameterized.named_parameters(
      dict(testcase_name="map", program=rasp.Map(lambda x: x + 1, rasp.tokens)),
      dict(
          testcase_name="sequence_map",
          program=rasp.SequenceMap(
              lambda x, y: x + y, rasp.tokens, rasp.indices
          ),
      ),
      dict(
          testcase_name="sequence_map_with_same_input",
          program=rasp.SequenceMap(
              lambda x, y: x + y, rasp.tokens, rasp.indices
          ),
      ),
      dict(
          testcase_name="select_aggregate",
          program=rasp.Aggregate(
              rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
              rasp.Map(lambda x: 1, rasp.tokens),
          ),
      ),
  )
  def test_rasp_program_and_transformer_produce_same_output(self, program):
    vocab = {0, 1, 2}
    max_seq_len = 3
    assembled_model = compiling.compile_rasp_to_model(
        program, vocab, max_seq_len, compiler_bos=_COMPILER_BOS
    )

    test_outputs = {}
    rasp_outputs = {}
    for val in vocab:
      test_outputs[val] = assembled_model.apply([_COMPILER_BOS, val]).decoded[1]
      rasp_outputs[val] = program([val])[0]

    with self.subTest(val=0):
      self.assertEqual(test_outputs[0], rasp_outputs[0])
    with self.subTest(val=1):
      self.assertEqual(test_outputs[1], rasp_outputs[1])
    with self.subTest(val=2):
      self.assertEqual(test_outputs[2], rasp_outputs[2])

  @parameterized.named_parameters(*test_cases.TEST_CASES)
  def test_compiled_models_produce_expected_output(
      self, program, vocab, test_input, expected_output, max_seq_len, **kwargs
  ):
    del kwargs
    assembled_model = compiling.compile_rasp_to_model(
        program, vocab, max_seq_len, compiler_bos=_COMPILER_BOS
    )
    test_output = assembled_model.apply([_COMPILER_BOS] + test_input)

    if isinstance(expected_output[0], (int, float)):
      np.testing.assert_allclose(
          test_output.decoded[1:], expected_output, atol=1e-7, rtol=0.005
      )
    else:
      self.assertSequenceEqualWhenExpectedIsNotNone(
          test_output.decoded[1:], expected_output
      )

  @parameterized.named_parameters(*test_cases.CAUSAL_TEST_CASES)
  def test_compiled_causal_models_produce_expected_output(
      self, program, vocab, test_input, expected_output, max_seq_len, **kwargs
  ):
    del kwargs
    assembled_model = compiling.compile_rasp_to_model(
        program,
        vocab,
        max_seq_len,
        causal=True,
        compiler_bos=_COMPILER_BOS,
        compiler_pad=_COMPILER_PAD,
    )
    test_output = assembled_model.apply([_COMPILER_BOS] + test_input)

    if isinstance(expected_output[0], (int, float)):
      np.testing.assert_allclose(
          test_output.decoded[1:], expected_output, atol=1e-7, rtol=0.005
      )
    else:
      self.assertSequenceEqualWhenExpectedIsNotNone(
          test_output.decoded[1:], expected_output
      )

  @parameterized.named_parameters(*test_cases.UNSUPPORTED_TEST_CASES)
  def test_unsupported_programs_raise_exception(
      self, program, vocab, max_seq_len
  ):
    with self.assertRaises(NotImplementedError):
      compiling.compile_rasp_to_model(
          program,
          vocab,
          max_seq_len,
          causal=True,
          compiler_bos=_COMPILER_BOS,
          compiler_pad=_COMPILER_PAD,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="reverse_1",
          program=lib.make_reverse(rasp.tokens),
          vocab={"a", "b", "c", "d"},
          test_input=list("abcd"),
          expected_output=list("dcba"),
          max_seq_len=5,
      ),
      dict(
          testcase_name="reverse_2",
          program=lib.make_reverse(rasp.tokens),
          vocab={"a", "b", "c", "d"},
          test_input=list("abc"),
          expected_output=list("cba"),
          max_seq_len=5,
      ),
      dict(
          testcase_name="reverse_3",
          program=lib.make_reverse(rasp.tokens),
          vocab={"a", "b", "c", "d"},
          test_input=list("ad"),
          expected_output=list("da"),
          max_seq_len=5,
      ),
      dict(
          testcase_name="reverse_4",
          program=lib.make_reverse(rasp.tokens),
          vocab={"a", "b", "c", "d"},
          test_input=["c"],
          expected_output=["c"],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_categorical_1",
          program=rasp.categorical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=list("abc"),
          expected_output=[3, 3, 3],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_categorical_2",
          program=rasp.categorical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=list("ad"),
          expected_output=[2, 2],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_categorical_3",
          program=rasp.categorical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=["c"],
          expected_output=[1],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_numerical_1",
          program=rasp.numerical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=list("abc"),
          expected_output=[3, 3, 3],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_numerical_2",
          program=rasp.numerical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=list("ad"),
          expected_output=[2, 2],
          max_seq_len=5,
      ),
      dict(
          testcase_name="length_numerical_3",
          program=rasp.numerical(lib.make_length()),
          vocab={"a", "b", "c", "d"},
          test_input=["c"],
          expected_output=[1],
          max_seq_len=5,
      ),
  )
  def test_compiled_models_produce_expected_output_with_padding(
      self, program, vocab, test_input, expected_output, max_seq_len, **kwargs
  ):
    del kwargs
    assembled_model = compiling.compile_rasp_to_model(
        program,
        vocab,
        max_seq_len,
        compiler_bos=_COMPILER_BOS,
        compiler_pad=_COMPILER_PAD,
    )

    pad_len = max_seq_len - len(test_input)
    test_input = test_input + [_COMPILER_PAD] * pad_len
    test_input = [_COMPILER_BOS] + test_input
    test_output = assembled_model.apply(test_input)
    output = test_output.decoded
    output_len = len(output)
    output_stripped = test_output.decoded[1 : output_len - pad_len]

    self.assertEqual(output[0], _COMPILER_BOS)
    if isinstance(expected_output[0], (int, float)):
      np.testing.assert_allclose(
          output_stripped, expected_output, atol=1e-7, rtol=0.005
      )
    else:
      self.assertEqual(output_stripped, expected_output)

  @parameterized.named_parameters(
      dict(
          testcase_name="threshold",
          make_program=lambda frac: rasp.Map(lambda x: x > 0.5, frac),
      ),
      dict(
          testcase_name="scale",
          make_program=lambda frac: rasp.numerical(
              rasp.Map(lambda x: x * 10, frac)
          ),
      ),
  )
  def test_map_after_numerical_aggregate_matches_rasp_exhaustively(
      self, make_program
  ):
    # A Map on a mean of binary values must see every value k / n, not only
    # 1 / n, otherwise e.g. 2/3 is snapped to 1/2.
    max_seq_len = 5
    program = make_program(_make_fraction_of_ones())
    model = compiling.compile_rasp_to_model(
        program, {0, 1}, max_seq_len, compiler_bos=_COMPILER_BOS
    )
    for length in range(1, max_seq_len + 1):
      for tokens in itertools.product([0, 1], repeat=length):
        tokens = list(tokens)
        with self.subTest(tokens=tokens):
          actual = model.apply([_COMPILER_BOS] + tokens).decoded[1:]
          np.testing.assert_allclose(
              np.asarray(actual, dtype=float),
              np.asarray(program(tokens), dtype=float),
              atol=1e-3,
          )

  @parameterized.parameters(10, 20)
  def test_map_after_numerical_aggregate_matches_rasp_on_long_sequences(
      self, max_seq_len
  ):
    # With many distinct means the gaps between them get small, so the MLP has
    # to be sharper than the default mlp_exactness to tell them apart.
    program = rasp.numerical(
        rasp.Map(lambda x: x * 10, _make_fraction_of_ones())
    )
    model = compiling.compile_rasp_to_model(
        program, {0, 1}, max_seq_len, compiler_bos=_COMPILER_BOS
    )
    for tokens in _sample_binary_inputs(max_seq_len, 30, seed=0):
      with self.subTest(tokens=tokens):
        actual = model.apply([_COMPILER_BOS] + tokens).decoded[1:]
        np.testing.assert_allclose(
            np.asarray(actual, dtype=float),
            np.asarray(program(tokens), dtype=float),
            atol=1e-3,
        )

  @parameterized.parameters(({0},), ({1},))
  def test_map_after_numerical_aggregate_of_single_value_vocab(self, vocab):
    # Selecting strictly earlier positions leaves the first position without
    # any selection, so the default 0 is output next to the mean (0 or 1).
    max_seq_len = 4
    is_one = rasp.numerical(rasp.Map(lambda t: t == 1, rasp.tokens))
    earlier = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LT)
    mean = rasp.numerical(rasp.Aggregate(earlier, is_one, default=0))
    program = rasp.numerical(rasp.Map(lambda x: x * 10, mean))
    model = compiling.compile_rasp_to_model(
        program, vocab, max_seq_len, compiler_bos=_COMPILER_BOS
    )
    for length in range(1, max_seq_len + 1):
      tokens = [next(iter(vocab))] * length
      with self.subTest(tokens=tokens):
        actual = model.apply([_COMPILER_BOS] + tokens).decoded[1:]
        np.testing.assert_allclose(actual, program(tokens), atol=1e-3)


if __name__ == "__main__":
  absltest.main()

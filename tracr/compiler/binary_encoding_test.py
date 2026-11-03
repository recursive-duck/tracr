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
"""Compiling programs whose S-Ops use the binary (bit-coded) encoding."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import shrinking
from tracr.compiler import validating
from tracr.rasp import rasp

_BOS = "binary_encoding_test_BOS"
_PAD = "binary_encoding_test_PAD"
_VOCAB = (0, 1, 2, 3)
_MAX_SEQ_LEN = 3

jax.config.update("jax_default_matmul_precision", "float32")

EQ = rasp.Comparison.EQ


def _all_inputs(vocab, max_seq_len):
  return [list(xs) for n in range(1, max_seq_len + 1)
          for xs in itertools.product(vocab, repeat=n)]


def _distinct_inputs(vocab, max_seq_len):
  return [list(xs) for n in range(1, max_seq_len + 1)
          for xs in itertools.permutations(vocab, n)]


# Each builder takes the encoding to give tokens, indices and every
# intermediate S-Op, so the same program compiles one-hot and bit-coded.
def _map(enc):
  return enc(rasp.Map(lambda x: (3 * x + 1) % 5, enc(rasp.tokens)))


def _map_to_categorical(enc):
  return rasp.categorical(rasp.Map(lambda x: x % 3, enc(rasp.tokens)))


def _map_from_categorical(enc):
  return enc(rasp.Map(lambda x: x * x, rasp.categorical(rasp.tokens)))


def _map_to_numerical(enc):
  return rasp.numerical(rasp.Map(lambda x: x / 4, enc(rasp.tokens)))


def _sequence_map(enc):
  return enc(rasp.SequenceMap(lambda x, y: x + 2 * y, enc(rasp.tokens),
                              enc(rasp.indices)))


def _sequence_map_mixed(enc):
  return enc(rasp.SequenceMap(lambda x, y: x * y, enc(rasp.tokens),
                              rasp.categorical(rasp.indices)))


def _aggregate_by_index(enc):
  return enc(rasp.Aggregate(
      rasp.Select(enc(rasp.indices), enc(rasp.indices), EQ), enc(rasp.tokens)))


def _reverse(enc):
  opp = enc(rasp.Map(lambda i: _MAX_SEQ_LEN - 1 - i, enc(rasp.indices)))
  return enc(rasp.Aggregate(rasp.Select(enc(rasp.indices), opp, EQ),
                            enc(rasp.tokens)))


def _histogram(enc):
  return enc(rasp.SelectorWidth(rasp.Select(enc(rasp.tokens),
                                            enc(rasp.tokens), EQ)))


def _histogram_categorical_out(enc):
  return rasp.categorical(rasp.SelectorWidth(
      rasp.Select(enc(rasp.tokens), enc(rasp.tokens), EQ)))


def _and_with_categorical_term(enc):
  sel = rasp.SelectorAnd(
      rasp.Select(enc(rasp.tokens), enc(rasp.tokens), EQ),
      rasp.Select(rasp.categorical(rasp.Map(lambda x: x, rasp.indices)),
                  rasp.categorical(rasp.Map(lambda x: x, rasp.indices)),
                  rasp.Comparison.LEQ))
  return rasp.categorical(rasp.SelectorWidth(sel))


def _numerical_mean(enc):
  is_zero = rasp.numerical(rasp.Map(lambda x: 1 if x == 0 else 0,
                                    enc(rasp.tokens)))
  return rasp.numerical(rasp.Aggregate(
      rasp.Select(enc(rasp.tokens), enc(rasp.tokens), EQ), is_zero,
      default=0))


def _numerical_to_binary(enc):
  frac = rasp.numerical(rasp.Aggregate(
      rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ),
      rasp.numerical(rasp.Map(lambda x: 1 if x > 1 else 0, rasp.tokens)),
      default=0))
  return enc(rasp.Map(lambda x: round(6 * x), frac))


def _sort_by_token(enc):
  # Tokens are distinct: move each token to the position given by its rank.
  # LT needs categorical keys and queries, so read the tokens into one-hot.
  onehot = rasp.categorical(rasp.Map(lambda x: x, enc(rasp.tokens)))
  rank = enc(rasp.SelectorWidth(rasp.Select(onehot, onehot,
                                            rasp.Comparison.LT)))
  return enc(rasp.Aggregate(rasp.Select(rank, enc(rasp.indices), EQ),
                            enc(rasp.tokens)))


_PROGRAMS = [
    ("map", _map, _all_inputs),
    ("map_to_categorical", _map_to_categorical, _all_inputs),
    ("map_from_categorical", _map_from_categorical, _all_inputs),
    ("map_to_numerical", _map_to_numerical, _all_inputs),
    ("sequence_map", _sequence_map, _all_inputs),
    ("sequence_map_mixed", _sequence_map_mixed, _all_inputs),
    ("aggregate_by_index", _aggregate_by_index, _all_inputs),
    ("reverse", _reverse, _all_inputs),
    ("histogram", _histogram, _all_inputs),
    ("histogram_categorical_out", _histogram_categorical_out, _all_inputs),
    ("and_with_categorical_term", _and_with_categorical_term, _all_inputs),
    ("numerical_mean", _numerical_mean, _all_inputs),
    ("numerical_to_binary", _numerical_to_binary, _all_inputs),
    ("sort_by_token", _sort_by_token, _distinct_inputs),
]


def _compile(program, max_seq_len=_MAX_SEQ_LEN):
  return compiling.compile_rasp_to_model(
      program, set(_VOCAB), max_seq_len, compiler_bos=_BOS, compiler_pad=_PAD)


def _assert_outputs_equal(test, actual, expected, xs):
  for a, e in zip(actual, expected):
    if e is None:
      continue
    if isinstance(e, float):
      test.assertAlmostEqual(a, e, places=3, msg=f"input {xs}")
    else:
      test.assertEqual(a, e, msg=f"input {xs}: {actual} != {expected}")


class BinaryEncodingTest(parameterized.TestCase):

  @parameterized.named_parameters(*_PROGRAMS)
  def test_compiled_binary_program_matches_rasp(self, build, inputs):
    program = build(rasp.binary)
    model = _compile(program)
    for xs in inputs(_VOCAB, _MAX_SEQ_LEN):
      out = model.apply([_BOS] + xs).decoded[1:]
      _assert_outputs_equal(self, out, program(xs), xs)

  @parameterized.named_parameters(*_PROGRAMS)
  def test_binary_and_categorical_decode_the_same(self, build, inputs):
    # Positions where RASP gives None decode arbitrarily; skip them.
    program = build(rasp.binary)
    binary_model = _compile(program)
    categorical_model = _compile(build(rasp.categorical))
    for xs in inputs(_VOCAB, _MAX_SEQ_LEN):
      defined = [y is not None for y in program(xs)]
      expected = categorical_model.apply([_BOS] + xs).decoded[1:]
      actual = binary_model.apply([_BOS] + xs).decoded[1:]
      _assert_outputs_equal(
          self, actual,
          [e if d else None for e, d in zip(expected, defined)], xs)

  def test_binary_tokens_use_fewer_dims(self):
    binary_model = _compile(_map(rasp.binary))
    categorical_model = _compile(_map(rasp.categorical))
    self.assertLess(len(binary_model.residual_labels),
                    len(categorical_model.residual_labels))
    self.assertIn("tokens:bit00", binary_model.residual_labels)
    self.assertNotIn("tokens:0", binary_model.residual_labels)

  def test_padded_input_matches_rasp(self):
    program = _reverse(rasp.binary)
    model = _compile(program, max_seq_len=_MAX_SEQ_LEN + 1)
    xs = [1, 3, 2]
    out = model.apply([_BOS] + xs + [_PAD]).decoded[1:1 + len(xs)]
    expected = rasp.Aggregate(
        rasp.Select(rasp.indices,
                    rasp.Map(lambda i: _MAX_SEQ_LEN - 1 - i, rasp.indices),
                    EQ), rasp.tokens)(xs)
    _assert_outputs_equal(self, out, expected, xs)

  def test_shrunk_binary_model_matches(self):
    program = _histogram(rasp.binary)
    model = _compile(program)
    shrunk = shrinking.shrink(model, reuse=False)
    for xs in _all_inputs(_VOCAB, _MAX_SEQ_LEN):
      self.assertEqual(shrunk.apply([_BOS] + xs)[1:], program(xs))


class BinaryValidationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("less_than", rasp.Comparison.LT, "Comparison.EQ"),
      ("true", rasp.Comparison.TRUE, "Comparison.EQ"),
  )
  def test_non_eq_predicate_on_binary_is_rejected(self, predicate, message):
    tokens = rasp.binary(rasp.tokens)
    program = rasp.SelectorWidth(rasp.Select(tokens, tokens, predicate))
    issues = validating.static_validate(program)
    self.assertLen(issues, 1)
    self.assertIn(message, issues[0].reason)
    with self.assertRaisesRegex(NotImplementedError, message):
      _compile(program)

  def test_mixed_binary_and_categorical_select_is_rejected(self):
    tokens = rasp.binary(rasp.tokens)
    program = rasp.SelectorWidth(rasp.Select(
        tokens, rasp.categorical(rasp.Map(lambda x: x, tokens)), EQ))
    issues = validating.static_validate(program)
    self.assertLen(issues, 1)
    self.assertIn("binary keys and queries", issues[0].reason)

  def test_tokens_with_two_encodings_are_rejected(self):
    program = rasp.SequenceMap(lambda x, y: x + y, rasp.binary(rasp.tokens),
                               rasp.Map(lambda x: x, rasp.tokens))
    issues = validating.static_validate(program)
    self.assertLen(issues, 1)
    self.assertIn("several encodings", issues[0].reason)

  def test_binary_aggregate_with_default_is_rejected(self):
    tokens = rasp.binary(rasp.tokens)
    program = rasp.binary(rasp.Aggregate(rasp.Select(tokens, tokens, EQ),
                                         tokens, default=0))
    issues = validating.static_validate(program)
    self.assertLen(issues, 1)
    self.assertIn("None as default", issues[0].reason)

  def test_binary_codes_are_shared_across_a_select(self):
    # Queries and keys with different value sets must still agree bit-wise.
    indices = rasp.binary(rasp.indices)
    shifted = rasp.binary(rasp.Map(lambda i: i + 1, indices))
    program = rasp.binary(rasp.Aggregate(rasp.Select(indices, shifted, EQ),
                                         rasp.binary(rasp.tokens)))
    model = _compile(program)
    for xs in _all_inputs(_VOCAB, _MAX_SEQ_LEN):
      out = model.apply([_BOS] + xs).decoded[1:]
      _assert_outputs_equal(self, out, program(xs), xs)


if __name__ == "__main__":
  absltest.main()

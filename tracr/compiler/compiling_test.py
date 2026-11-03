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
"""Regression tests for the full RASP -> transformer compilation."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk
import jax
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.rasp import rasp

# Force float32 precision on TPU, which otherwise defaults to float16.
jax.config.update("jax_default_matmul_precision", "float32")


class ValueSizeLargerThanKeySizeTest(parameterized.TestCase):
  """Heads whose value width exceeds their query/key width must compile."""

  @parameterized.parameters(5, 6, 7, 8, 20)
  def test_aggregate_over_tokens_matches_rasp(self, vocab_size):
    # Query/key space is indices (size max_seq_len + 1 at most), value space is
    # the tokens, so for large vocabularies value_size > query/key size.
    max_seq_len = 3
    program = rasp.Aggregate(
        rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ),
        rasp.tokens)
    model = compiling.compile_rasp_to_model(
        program, set(range(vocab_size)), max_seq_len)

    last = vocab_size - 1
    test_inputs = [[0], [last], [0, last], [last, 1], [last, 0, 1],
                   [last, last - 1, last - 2]]
    for test_input in test_inputs:
      with self.subTest(test_input=test_input):
        actual = model.apply([compiling.COMPILER_BOS] + test_input).decoded[1:]
        self.assertEqual(actual, program(test_input))


class AttentionLogitScaleTest(parameterized.TestCase):
  """Compiled logits must equal coldness * craft score at any key size."""

  @parameterized.parameters(3, 40)
  def test_logits_match_craft_scores(self, vocab_size):
    coldness, bos_score, match_score = 100., 0.5, 1.
    program = rasp.Aggregate(
        rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ),
        rasp.tokens)
    model = compiling.compile_rasp_to_model(
        program, set(range(vocab_size)), max_seq_len=3)
    self.assertGreater(model.model_config.key_size, 1)
    test_input = [0, 1, vocab_size - 1]
    out = model.apply([compiling.COMPILER_BOS] + test_input)
    logits = np.asarray(out.attn_logits[0])[0, 0]
    for query in range(1, len(test_input) + 1):
      self.assertAlmostEqual(
          float(logits[query, 0]), coldness * bos_score, places=3)
      self.assertAlmostEqual(
          float(logits[query, query]), coldness * match_score, places=3)


_VOCAB = {1, 2, 3}
_MAX_SEQ_LEN = 5
_TEST_BOS = "compiling_test_BOS"
_TEST_PAD = compiling.COMPILER_PAD


def _is_one(sop):
  return rasp.Map(lambda t: t == 1, sop)


class PaddingTest(parameterized.TestCase):
  """Output at non-PAD positions must not depend on trailing PAD tokens."""

  def _assert_same_with_and_without_padding(self, model, tokens):
    unpadded = model.apply([_TEST_BOS] + tokens)
    for num_pads in range(1, _MAX_SEQ_LEN - len(tokens) + 1):
      with self.subTest(tokens=tokens, num_pads=num_pads):
        padded = model.apply([_TEST_BOS] + tokens + [_TEST_PAD] * num_pads)
        num_real = len(tokens) + 1  # Including BOS.
        self.assertEqual(
            [type(x) for x in padded.decoded[:num_real]],
            [type(x) for x in unpadded.decoded])
        for actual, expected in zip(padded.decoded[:num_real],
                                    unpadded.decoded):
          if isinstance(expected, str):
            self.assertEqual(actual, expected)
          else:
            self.assertAlmostEqual(actual, expected, places=3)
        # PAD positions must not poison the model with NaN/inf.
        for array in (padded.transformer_output, *padded.layer_outputs,
                      *padded.attn_logits):
          self.assertTrue(np.isfinite(np.asarray(array)).all())

  @parameterized.named_parameters(
      dict(
          testcase_name="indices_selector_width",
          program=rasp.SequenceMap(
              lambda t, n: n, rasp.tokens,
              rasp.SelectorWidth(
                  rasp.Select(rasp.indices, rasp.indices,
                              rasp.Comparison.TRUE)))),
      dict(
          testcase_name="indices_geq_mean",
          program=rasp.numerical(
              rasp.Aggregate(
                  rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.GEQ),
                  rasp.numerical(_is_one(rasp.tokens)),
                  default=0))),
      dict(testcase_name="length", program=lib.make_length()),
      dict(testcase_name="reverse", program=lib.make_reverse(rasp.tokens)),
      dict(
          testcase_name="frac_prevs",
          program=lib.make_frac_prevs(_is_one(rasp.tokens))),
  )
  def test_padding_does_not_change_non_pad_output(self, program):
    model = compiling.compile_rasp_to_model(
        program, _VOCAB, _MAX_SEQ_LEN, compiler_bos=_TEST_BOS)
    for length in range(1, 4):
      for tokens in itertools.product(sorted(_VOCAB), repeat=length):
        self._assert_same_with_and_without_padding(model, list(tokens))

  @parameterized.named_parameters(
      dict(
          testcase_name="indices_geq_mean",
          program=rasp.numerical(
              rasp.Aggregate(
                  rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.GEQ),
                  rasp.numerical(_is_one(rasp.tokens)),
                  default=0))),
      dict(
          testcase_name="frac_prevs",
          program=lib.make_frac_prevs(_is_one(rasp.tokens))),
  )
  def test_padding_does_not_change_non_pad_output_causal(self, program):
    model = compiling.compile_rasp_to_model(
        program, _VOCAB, _MAX_SEQ_LEN, causal=True, compiler_bos=_TEST_BOS)
    for tokens in ([1], [1, 1], [2, 1, 3], [1, 2, 1, 1]):
      self._assert_same_with_and_without_padding(model, tokens)

  def test_pad_token_of_model_matches_input_encoder(self):
    model = compiling.compile_rasp_to_model(
        rasp.Map(lambda t: t + 1, rasp.tokens),
        _VOCAB,
        _MAX_SEQ_LEN,
        compiler_bos=_TEST_BOS)
    pad_id = hk.without_apply_rng(
        hk.transform(lambda: model.get_compiled_model().pad_token)).apply(
            model.params)
    self.assertIsNotNone(model.input_encoder.pad_encoding)
    self.assertEqual(pad_id, model.input_encoder.pad_encoding)
    self.assertEqual(
        model.input_encoder.encode([_TEST_BOS, _TEST_PAD])[1], pad_id)


if __name__ == "__main__":
  absltest.main()

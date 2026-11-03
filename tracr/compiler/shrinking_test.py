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
"""Tests for compiler.shrinking."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import lib
from tracr.compiler import shrinking
from tracr.rasp import rasp

_BOS = "BOS"
_MAX_SEQ_LEN = 4
_BATCH = 512
_INT_VOCAB = {0, 1, 2, 3}
_BRACKETS = {"(", ")", "{", "}"}
# Float32 sums of the same nonzero terms in another order.
_NUMERICAL_ATOL = 1e-5
# Top-2 score margin under which the original's argmax is a rounding tie.
_TIE_TOL = 1e-5


def _programs():
  return {
      "length": (lambda: lib.make_length(), _INT_VOCAB),
      "reverse": (lambda: lib.make_reverse(rasp.tokens), _INT_VOCAB),
      "hist": (lambda: lib.make_hist(), _INT_VOCAB),
      "sort_unique": (lambda: lib.make_sort_unique(rasp.tokens, rasp.tokens),
                      _INT_VOCAB),
      "sort": (lambda: lib.make_sort(rasp.tokens, rasp.tokens,
                                     max_seq_len=_MAX_SEQ_LEN, min_key=1),
               _INT_VOCAB),
      "sort_freq": (lambda: lib.make_sort_freq(max_seq_len=_MAX_SEQ_LEN),
                    _INT_VOCAB),
      "frac_prevs": (lambda: lib.make_frac_prevs(rasp.tokens == "("),
                     _BRACKETS),
      "pair_balance": (lambda: lib.make_pair_balance(rasp.tokens, "(", ")"),
                       _BRACKETS),
      "shuffle_dyck2": (lambda: lib.make_shuffle_dyck2(), _BRACKETS),
      "count_less_freq": (lambda: lib.make_count_less_freq(n=2), _INT_VOCAB),
  }


def _all_inputs(vocab, max_len):
  return [list(x) for n in range(1, max_len + 1)
          for x in itertools.product(sorted(vocab), repeat=n)]


class ShrinkTest(parameterized.TestCase):

  @parameterized.product(name=list(_programs()), reuse=[False, True],
                         causal=[False, True])
  def test_matches_original_on_every_input(self, name, reuse, causal):
    make, vocab = _programs()[name]
    model = compiling.compile_rasp_to_model(
        make(), vocab=vocab, max_seq_len=_MAX_SEQ_LEN, compiler_bos=_BOS,
        causal=causal)
    shrunk = shrinking.shrink(model, reuse=reuse)
    inputs = _all_inputs(vocab, _MAX_SEQ_LEN)
    result = shrinking.compare(model, shrunk, inputs, batch_size=_BATCH,
                               tie_tol=_TIE_TOL)
    self.assertEqual(result.sequences, len(inputs))
    if shrunk.categorical_output:
      self.assertEqual(result.decoded_mismatches, 0)
    self.assertLessEqual(result.max_abs_diff, _NUMERICAL_ATOL)

  def test_compare_separates_ties_from_decided_mismatches(self):
    model = compiling.compile_rasp_to_model(
        lib.make_hist(), vocab=_INT_VOCAB, max_seq_len=_MAX_SEQ_LEN,
        compiler_bos=_BOS)
    shrunk = shrinking.shrink(model)
    inputs = _all_inputs(_INT_VOCAB, _MAX_SEQ_LEN)
    clean = shrinking.compare(model, shrunk, inputs, _BATCH, tie_tol=_TIE_TOL)
    self.assertEqual((clean.decoded_mismatches, clean.tie_flips), (0, 0))
    # Rolling the output columns by one moves every decided argmax.
    shrunk.unembed_matrix = np.roll(shrunk.unembed_matrix, 1, axis=1)
    broken = shrinking.compare(model, shrunk, inputs, _BATCH, tie_tol=_TIE_TOL)
    self.assertEqual(broken.decoded_mismatches, len(inputs))

  @parameterized.parameters(False, True)
  def test_apply_matches_assembled_apply(self, reuse):
    model = compiling.compile_rasp_to_model(
        lib.make_sort_freq(max_seq_len=_MAX_SEQ_LEN), vocab=_INT_VOCAB,
        max_seq_len=_MAX_SEQ_LEN, compiler_bos=_BOS)
    shrunk = shrinking.shrink(model, reuse=reuse)
    for x in ([3, 1, 1, 2], [0], [2, 2, 2]):
      self.assertEqual(shrunk.apply([_BOS] + x), model.apply([_BOS] + x).decoded)

  @parameterized.parameters(False, True)
  def test_never_grows_and_drops_padding(self, reuse):
    model = compiling.compile_rasp_to_model(
        lib.make_shuffle_dyck2(), vocab=_BRACKETS, max_seq_len=_MAX_SEQ_LEN,
        compiler_bos=_BOS)
    shrunk = shrinking.shrink(model, reuse=reuse)
    stats = shrunk.stats
    self.assertLess(stats.params_after, stats.params_before)
    self.assertLessEqual(stats.d_model_after, stats.d_model_before)
    self.assertLess(stats.heads_after, stats.heads_before)
    for layer in shrunk.layers:
      self.assertLen(layer.qk_sizes, len(layer.v_sizes))
      self.assertTrue(all(v > 0 for v in layer.v_sizes))

  def test_reuse_narrows_residual_with_two_erase_units_per_reused_dim(self):
    model = compiling.compile_rasp_to_model(
        lib.make_shuffle_dyck2(), vocab=_BRACKETS, max_seq_len=_MAX_SEQ_LEN,
        compiler_bos=_BOS)
    trimmed = shrinking.shrink(model, reuse=False)
    reused = shrinking.shrink(model, reuse=True)
    self.assertEqual(trimmed.stats.dims_reused, 0)
    self.assertGreater(reused.stats.dims_reused, 0)
    self.assertEqual(reused.stats.d_model_after,
                     trimmed.stats.d_model_after - reused.stats.dims_reused)
    self.assertEqual(reused.stats.erase_units, 2 * reused.stats.dims_reused)
    self.assertEqual(sum(len(s) for s in reused.slot_labels),
                     reused.stats.dims_kept)

  def test_rejects_nonzero_bias(self):
    model = compiling.compile_rasp_to_model(
        lib.make_length(), vocab=_INT_VOCAB, max_seq_len=_MAX_SEQ_LEN,
        compiler_bos=_BOS)
    name = "transformer/layer_0/mlp/linear_1"
    model.params[name]["b"] = jax.numpy.ones_like(model.params[name]["b"])
    with self.assertRaisesRegex(ValueError, "zero biases"):
      shrinking.shrink(model)

  def test_all_zero_head_columns_give_uniform_attention(self):
    # length attends to every position with equal weight: its query/key may
    # be trimmed to zero columns, which must still produce uniform weights.
    model = compiling.compile_rasp_to_model(
        lib.make_length(), vocab=_INT_VOCAB, max_seq_len=_MAX_SEQ_LEN,
        compiler_bos=_BOS)
    shrunk = shrinking.shrink(model)
    out = shrunk.apply([_BOS, 1, 2, 3])
    self.assertEqual(out, model.apply([_BOS, 1, 2, 3]).decoded)
    np.testing.assert_array_equal(out[1:], [3, 3, 3])


if __name__ == "__main__":
  absltest.main()

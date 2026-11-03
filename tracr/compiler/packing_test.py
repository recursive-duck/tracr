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
"""Tests for compiler.packing."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import packing
from tracr.compiler import shrinking
from tracr.compiler import shrinking_test

_BOS = "BOS"
_MAX_SEQ_LEN = shrinking_test._MAX_SEQ_LEN
_BATCH = shrinking_test._BATCH
_NUMERICAL_ATOL = shrinking_test._NUMERICAL_ATOL
_TIE_TOL = shrinking_test._TIE_TOL
# Small mixed set: numerical and categorical, int and bracket vocab.
_SMALL = ("frac_prevs", "hist", "shuffle_dyck2")


@functools.lru_cache(maxsize=None)
def _compiled(name, causal=False, max_seq_len=_MAX_SEQ_LEN):
  make, vocab = shrinking_test._programs()[name]
  model = compiling.compile_rasp_to_model(
      make(), vocab=vocab, max_seq_len=max_seq_len, compiler_bos=_BOS,
      causal=causal)
  return model, vocab


def _packed_ids(packed, task, inputs):
  """Every input of a task as packed token ids, padded to max_seq_len + 1."""
  t = packed.tasks[task]
  enc = t.input_encoder
  ids = np.full((len(inputs), _MAX_SEQ_LEN + 1), enc.pad_encoding, np.int32)
  for r, x in enumerate(inputs):
    encoded = enc.encode([enc.bos_token] + list(x))
    ids[r, :len(encoded)] = encoded
  return ids + t.token_offset


def _preactivations(packed, ids):
  """MLP unit inputs of every layer, [B, T, units] each, in float64."""
  seq_len = ids.shape[1]
  x = packed.token_embed[ids].astype(np.float64) + packed.pos_embed[:seq_len]
  mask = np.repeat(~packed.is_pad[ids][:, None, :], seq_len, axis=1)
  if packed.causal:
    mask &= np.tril(np.ones((seq_len, seq_len), dtype=bool))[None]
  out = []
  for layer in packed.layers:
    q, k, v = x @ layer.wq, x @ layer.wk, x @ layer.wv
    qk_edges = np.cumsum((0,) + layer.qk_sizes)
    v_edges = np.cumsum((0,) + layer.v_sizes)
    heads = []
    for h, div in enumerate(layer.divisors):
      qs = slice(qk_edges[h], qk_edges[h + 1])
      logits = np.einsum("btd,bTd->btT", q[..., qs], k[..., qs]) / div
      logits = np.where(mask, logits, -np.inf)
      weights = np.exp(logits - logits.max(-1, keepdims=True))
      weights /= weights.sum(-1, keepdims=True)
      vs = slice(v_edges[h], v_edges[h + 1])
      heads.append(np.einsum("btT,bTd->btd", weights, v[..., vs]))
    if heads:
      x = x + np.concatenate(heads, axis=-1) @ layer.wo
    pre = x @ layer.w1
    out.append(pre)
    x = x + np.maximum(pre, 0) @ layer.w2
  return out


class PackTest(parameterized.TestCase):

  def assert_tasks_match(self, names, packed, causal=False):
    for i, name in enumerate(names):
      with self.subTest(name):
        model, vocab = _compiled(name, causal)
        inputs = shrinking_test._all_inputs(vocab, _MAX_SEQ_LEN)
        result = shrinking.compare(model, packed.task(i), inputs, _BATCH,
                                   tie_tol=_TIE_TOL)
        self.assertEqual(result.sequences, len(inputs))
        if model.categorical_output:
          self.assertEqual(result.decoded_mismatches, 0)
        self.assertLessEqual(result.max_abs_diff, _NUMERICAL_ATOL)

  @parameterized.parameters(False, True)
  def test_every_task_matches_its_program_on_every_input(self, causal):
    names = list(shrinking_test._programs())
    packed = packing.pack([_compiled(n, causal)[0] for n in names], names)
    self.assert_tasks_match(names, packed, causal)

  def test_shares_dims_within_bounds(self):
    names = list(shrinking_test._programs())
    packed = packing.pack([_compiled(n)[0] for n in names], names)
    stats = packed.stats
    inputs = stats.gate_slots + stats.position_slots
    self.assertEqual(stats.d_model, inputs + stats.slots_final)
    self.assertGreaterEqual(stats.slots_final, stats.slots_lower_bound)
    self.assertGreaterEqual(stats.d_model, max(stats.d_programs))
    self.assertLess(stats.d_model, sum(stats.d_programs))
    self.assertLess(stats.params, stats.params_block_diagonal)
    self.assertEqual(stats.params, packed.num_params)

  def test_single_program_is_its_trimmed_model(self):
    model, _ = _compiled("sort_freq")
    packed = packing.pack([model])
    trimmed = shrinking.shrink(model, reuse=False)
    self.assertEqual(packed.stats.d_model, trimmed.stats.d_model_after)
    self.assertEqual(packed.num_params, trimmed.num_params)

  def test_integer_program_repairs_a_bad_colouring(self):
    models = [_compiled(n)[0] for n in _SMALL]
    good = packing.pack(models, _SMALL)
    # One slot per value: the integer program must find a narrower layout.
    with mock.patch.object(packing, "_dsatur",
                           lambda adj: np.arange(len(adj))):
      packed = packing.pack(models, _SMALL)
    self.assertEqual(packed.stats.solver, "ilp")
    self.assertGreater(packed.stats.slots_dsatur, packed.stats.slots_final)
    self.assertLessEqual(packed.stats.slots_final, good.stats.slots_final)
    self.assertIn("Optimal", packed.stats.ilp_status)
    self.assert_tasks_match(_SMALL, packed)

  def test_without_gates_other_tasks_leak(self):
    names = list(shrinking_test._programs())
    packed = packing.pack([_compiled(n)[0] for n in names], names,
                          gate_scale=0.0, gate_offset=0.0)
    wrong = 0
    for i, name in enumerate(names):
      model, vocab = _compiled(name)
      inputs = shrinking_test._all_inputs(vocab, _MAX_SEQ_LEN)
      result = shrinking.compare(model, packed.task(i), inputs, _BATCH,
                                 tie_tol=_TIE_TOL)
      wrong += result.decoded_mismatches + (result.max_abs_diff
                                            > _NUMERICAL_ATOL)
    self.assertGreater(wrong, 0)

  @parameterized.parameters(False, True)
  def test_gates_shut_every_other_unit_on_every_input(self, onehot_bounds):
    names = list(shrinking_test._programs())
    packed = packing.pack([_compiled(n)[0] for n in names], names,
                          onehot_bounds=onehot_bounds)
    for i, name in enumerate(names):
      with self.subTest(name):
        inputs = shrinking_test._all_inputs(_compiled(name)[1], _MAX_SEQ_LEN)
        pre = _preactivations(packed, _packed_ids(packed, i, inputs))
        for layer, p in zip(packed.layers, pre):
          other = layer.unit_task != i
          if other.any():
            self.assertLess(p[..., other].max(), 0.0)

  def test_onehot_bounds_give_smaller_gates(self):
    names = list(shrinking_test._programs())
    models = [_compiled(n)[0] for n in names]
    loose = packing.pack(models, names, onehot_bounds=False)
    tight = packing.pack(models, names, onehot_bounds=True)
    self.assertLess(tight.stats.max_gate, loose.stats.max_gate)

  def test_rebalancing_keeps_outputs_bitwise_and_shrinks_weights(self):
    names = list(shrinking_test._programs())
    models = [_compiled(n)[0] for n in names]
    plain = packing.pack(models, names, rebalance=False)
    balanced = packing.pack(models, names, rebalance=True)
    self.assertLess(balanced.stats.max_weight, plain.stats.max_weight)
    for i, name in enumerate(names):
      with self.subTest(name):
        inputs = shrinking_test._all_inputs(_compiled(name)[1], _MAX_SEQ_LEN)
        ids = _packed_ids(plain, i, inputs) - plain.tasks[i].token_offset
        np.testing.assert_array_equal(
            np.asarray(plain.output_values(i, ids)),
            np.asarray(balanced.output_values(i, ids)))

  def test_rejects_different_lengths(self):
    with self.assertRaisesRegex(ValueError, "max_seq_len"):
      packing.pack([_compiled("hist")[0],
                    _compiled("hist", max_seq_len=_MAX_SEQ_LEN + 1)[0]])

  def test_rejects_different_causal_settings(self):
    with self.assertRaisesRegex(ValueError, "causal"):
      packing.pack([_compiled("hist")[0], _compiled("hist", causal=True)[0]])


if __name__ == "__main__":
  absltest.main()

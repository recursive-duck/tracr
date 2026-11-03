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

  def test_rejects_different_lengths(self):
    with self.assertRaisesRegex(ValueError, "max_seq_len"):
      packing.pack([_compiled("hist")[0],
                    _compiled("hist", max_seq_len=_MAX_SEQ_LEN + 1)[0]])

  def test_rejects_different_causal_settings(self):
    with self.assertRaisesRegex(ValueError, "causal"):
      packing.pack([_compiled("hist")[0], _compiled("hist", causal=True)[0]])


if __name__ == "__main__":
  absltest.main()

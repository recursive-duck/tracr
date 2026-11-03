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

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tracr.compiler import compiling
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


if __name__ == "__main__":
  absltest.main()

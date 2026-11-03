"""Tests for lib_copy: compiled exact copy == RASP == pure-Python reference."""

import collections
import itertools
import random

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tracr.compiler import compiling
from tracr.compiler import lib_copy
from tracr.rasp import rasp

jax.config.update("jax_default_matmul_precision", "float32")

_BOS = "lib_copy_test_BOS"
_SMALL_SLOTS = 3
_SMALL_BASE = 2
_SMALL_ID_LENS = (1, 2, 3)
_RASP_CHECK_EVERY = 7  # RASP is slow; check every 7th exhaustive case.
_RANDOM_LAYOUT = lib_copy.CopyLayout(id_len=6, num_slots=5, base=10)
_RANDOM_CASES = 150
_RANDOM_MAX_GAP = 3
_RANDOM_MAX_CALLS = 2
_SEED = 0


def _compile(program, layout, max_seq_len):
  return compiling.compile_rasp_to_model(
      program, layout.vocab, max_seq_len, compiler_bos=_BOS)


def _run(model, seqs):
  """Decoded outputs at real positions; one jitted batch per length."""
  enc, dec = model.input_encoder, model.output_encoder
  forward = jax.jit(lambda p, x: model.forward(p, x).unembedded_output)
  by_len = collections.defaultdict(list)
  for i, s in enumerate(seqs):
    by_len[len(s)].append(i)
  outs = [None] * len(seqs)
  for idx in by_len.values():
    ids = np.array([enc.encode([enc.bos_token] + seqs[i]) for i in idx],
                   dtype=np.int32)
    pred = np.asarray(forward(model.params, jnp.asarray(ids)))
    for i, row in zip(idx, pred):
      outs[i] = dec.decode(row.tolist())[1:]
  return outs


def _copy_mismatches(outs, refs):
  """Copy positions (reference not None) where the output differs."""
  return sum(o != r for out, ref in zip(outs, refs)
             for o, r in zip(out, ref) if r is not None)


def _exhaustive_cases(layout):
  """Every set of 1..num_slots sources (slots and order), every digit string,
  each source slot called once; copy positions hold the filler."""
  k = layout.id_len
  slots = range(layout.num_slots)
  for n in range(1, layout.num_slots + 1):
    for order in itertools.permutations(slots, n):
      for digits in itertools.product(range(layout.base), repeat=n * k):
        ids = {s: digits[i * k:(i + 1) * k] for i, s in enumerate(order)}
        for call in order:
          yield lib_copy.make_sequence(layout, ids, [call])


def _random_case(layout, rng):
  return lib_copy.random_sequence(layout, rng, _RANDOM_MAX_CALLS,
                                  _RANDOM_MAX_GAP)


class LibCopyTest(parameterized.TestCase):

  @parameterized.parameters(*_SMALL_ID_LENS)
  def test_exhaustive_small_cases_match_reference(self, k):
    layout = lib_copy.CopyLayout(id_len=k, num_slots=_SMALL_SLOTS,
                                 base=_SMALL_BASE)
    program = lib_copy.make_exact_copy(layout)
    seqs = list(_exhaustive_cases(layout))
    refs = [lib_copy.reference_copy(layout, s) for s in seqs]
    model = _compile(program, layout, max(map(len, seqs)))
    outs = _run(model, seqs)
    self.assertEqual(_copy_mismatches(outs, refs), 0)
    # One call per case, so k copy positions each.
    self.assertEqual(sum(r is not None for ref in refs for r in ref),
                     k * len(seqs))
    for seq, ref in list(zip(seqs, refs))[::_RASP_CHECK_EVERY]:
      self.assertEqual(rasp.evaluate(program, seq), ref, msg=f"{seq}")

  def test_random_larger_cases_match_reference(self):
    layout = _RANDOM_LAYOUT
    rng = random.Random(_SEED)
    seqs = [_random_case(layout, rng) for _ in range(_RANDOM_CASES)]
    self.assertLessEqual(
        max(map(len, seqs)),
        lib_copy.max_sequence_len(layout, _RANDOM_MAX_CALLS, _RANDOM_MAX_GAP))
    refs = [lib_copy.reference_copy(layout, s) for s in seqs]
    program = lib_copy.make_exact_copy(layout)
    model = _compile(program, layout, max(map(len, seqs)))
    self.assertEqual(_copy_mismatches(_run(model, seqs), refs), 0)
    for seq, ref in zip(seqs, refs):
      self.assertEqual(rasp.evaluate(program, seq), ref, msg=f"{seq}")

  def test_categorical_key_query_give_the_same_copies(self):
    layout = lib_copy.CopyLayout(id_len=2, num_slots=2, base=3)
    rng = random.Random(_SEED)
    seqs = [_random_case(layout, rng) for _ in range(_RANDOM_CASES)]
    refs = [lib_copy.reference_copy(layout, s) for s in seqs]
    program = lib_copy.make_exact_copy(layout, encoding=rasp.categorical)
    model = _compile(program, layout, max(map(len, seqs)))
    self.assertEqual(_copy_mismatches(_run(model, seqs), refs), 0)

  def test_copy_head_matches_on_bit_codes_in_two_attention_layers(self):
    layout = lib_copy.CopyLayout(id_len=3, num_slots=3, base=10)
    program = lib_copy.make_exact_copy(layout)
    model = _compile(program, layout, 2 * (layout.num_slots + 1) *
                     layout.period)
    # Labels get a unique suffix: copy_key_<id>:bit00.
    dims = {name: [l.split(":")[1] for l in model.residual_labels
                   if l.startswith(name + "_")]
            for name in ("copy_key", "copy_query")}
    for values in dims.values():
      self.assertNotEmpty(values)
      self.assertTrue(all(v.startswith("bit") for v in values), values)
    attn_layers = [
        l for l in range(model.model_config.num_layers)
        if np.abs(np.asarray(model.params[
            f"transformer/layer_{l}/attn/query"]["w"])).sum() > 0
    ]
    self.assertLen(attn_layers, 2)

  def test_docstring_example(self):
    layout = lib_copy.CopyLayout(id_len=2, num_slots=2, base=10)
    f = layout.filler
    seq = [f, layout.src(1), 4, 2, layout.src(0), 7, 7, layout.call(1), f, f]
    expected = [None] * 8 + [4, 2]
    self.assertEqual(lib_copy.reference_copy(layout, seq), expected)
    self.assertEqual(lib_copy.make_exact_copy(layout)(seq), expected)

  def test_truncated_call_copies_the_prefix(self):
    layout = lib_copy.CopyLayout(id_len=3, num_slots=1, base=10)
    seq = [layout.filler, layout.src(0), 5, 6, 7, layout.call(0), 5]
    expected = [None] * 6 + [5]
    self.assertEqual(lib_copy.reference_copy(layout, seq), expected)
    self.assertEqual(lib_copy.make_exact_copy(layout)(seq), expected)

  @parameterized.named_parameters(
      ("markers_too_close", lambda l: [l.src(0), 1, l.src(1), 1, 1]),
      ("two_sources", lambda l: [l.src(0), 1, 1, l.src(0), 0, 0]),
      ("short_source", lambda l: [l.src(0), 1]),
      ("source_not_digits", lambda l: [l.src(0), 1, l.filler]),
      ("call_without_source", lambda l: [l.src(0), 1, 1, l.call(1), 0, 0]),
      ("bad_token", lambda l: [l.filler + 1]),
  )
  def test_check_layout_rejects(self, build):
    layout = lib_copy.CopyLayout(id_len=2, num_slots=2, base=2)
    with self.assertRaises(ValueError):
      lib_copy.check_layout(layout, build(layout))

  def test_rejects_bad_layout(self):
    with self.assertRaises(ValueError):
      lib_copy.CopyLayout(id_len=0, num_slots=1, base=2)


if __name__ == "__main__":
  absltest.main()

"""Decoded model outputs vs the RASP program, for the shrink and pack reports.

The reports' `shrinking.compare` checks a shrunk or packed model against the
compiled original only, so a bug shared by both (as the sqrt(key_size) logit
divisor fixed in 5d345a6) shows 0 mismatches. This checks every model against
RASP itself, evaluated with rasp.fast_eval.
"""

import dataclasses
from typing import Any, Callable, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from tracr.rasp import fast_eval


@dataclasses.dataclass
class RaspCheck:
  sequences: int
  # Sequences with a position where the decoded output differs from RASP;
  # positions where RASP gives None are not checked.
  mismatches: int
  # Largest |output - RASP| over checked positions of numerical programs.
  max_abs_err: float


def expected_outputs(program, inputs: Sequence[Sequence[Any]]) -> List[List]:
  return fast_eval.evaluate_many(program, inputs)


def original_output_values(model) -> Callable[[np.ndarray], jax.Array]:
  """[B, T, D_out] outputs of an assembled model, like shrinking.compare."""
  forward = jax.jit(model.forward)
  unembed = jnp.asarray(model.unembed_matrix)
  return lambda ids: forward(model.params, jnp.asarray(ids)) \
      .transformer_output.output @ unembed


def check(output_values: Callable[[np.ndarray], Any], model,
          inputs: Sequence[Sequence[Any]], expected: Sequence[Sequence[Any]],
          batch_size: int, atol: float, rtol: float) -> RaspCheck:
  """Counts sequences whose decoded output differs from RASP.

  output_values maps token ids [B, T] (BOS first, PAD after the end) to
  output values [B, T, D_out]; `model` is the compiled original, which gives
  the encoders and the output type. Categorical outputs are argmax-decoded
  and must equal RASP; numerical ones must be within atol + rtol * |RASP|.
  """
  enc = model.input_encoder
  bos, pad_id = enc.bos_token, enc.pad_encoding
  categorical = model.categorical_output
  if categorical:
    labels = {i: v for v, i in model.output_encoder.encoding_map.items()}
  mismatches, max_err = 0, 0.0
  for start in range(0, len(inputs), batch_size):
    chunk = inputs[start:start + batch_size]
    want = expected[start:start + batch_size]
    width = 1 + max(len(x) for x in chunk)
    ids = np.full((len(chunk), width), pad_id, dtype=np.int32)
    for r, x in enumerate(chunk):
      encoded = enc.encode([bos] + list(x))
      ids[r, :len(encoded)] = encoded
    out = np.asarray(output_values(ids))
    if categorical:
      got = out.argmax(-1)
    else:
      assert out.shape[-1] == 1, out.shape
      got = out[..., 0]
    for r, row in enumerate(want):
      bad = False
      for i, e in enumerate(row):
        if e is None:
          continue
        g = got[r, i + 1]  # position 0 is BOS
        if categorical:
          bad |= labels.get(int(g)) != e
        else:
          err = abs(float(g) - float(e))
          max_err = max(max_err, err)
          bad |= not err <= atol + rtol * abs(float(e))
      mismatches += bad
  return RaspCheck(sequences=len(inputs), mismatches=int(mismatches),
                   max_abs_err=max_err)

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
"""Shrink an assembled model without changing what it computes.

The assembler pads every layer to the widest one and keeps every residual
dimension for the whole model. This module removes what cannot affect the
output, and can let residual dimensions be reused once their value is dead:

1. Trim. Drop residual dims that no kept block reads or no kept block writes,
   padding heads, query/key columns that are zero on one side, value columns
   with no effect, and MLP units with zero input or zero output weights.
   Removing one part can make another unused, so this runs to a fixed point.
2. Per-layer shapes. Each layer keeps only its own heads, head sizes and
   MLP units, instead of the maximum over all layers.
3. Reuse (optional). A dim whose last read has happened is erased by two MLP
   units, ReLU(x) and ReLU(-x), which write -x for x of any sign. Its slot can
   then hold a value born later. Slots are assigned greedily by birth time,
   which needs the fewest slots for interval lifetimes.

In exact arithmetic the shrunk model computes the same outputs. In floating
point, matmuls over different shapes may sum the same nonzero terms in a
different order, so numerical outputs can differ in the last bits.
"""

import dataclasses
import functools
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tracr.compiler import assemble
from tracr.transformer import encoder

# Value the original attention writes into masked logits.
_MASKED_LOGIT = -1e30


@dataclasses.dataclass
class ShrunkLayer:
  """One layer with its own head sizes and MLP width."""
  wq: np.ndarray  # [D, sum(qk_sizes)]
  wk: np.ndarray  # [D, sum(qk_sizes)]
  wv: np.ndarray  # [D, sum(v_sizes)]
  wo: np.ndarray  # [sum(v_sizes), D]
  qk_sizes: Tuple[int, ...]
  v_sizes: Tuple[int, ...]
  w1: np.ndarray  # [D, M]
  w2: np.ndarray  # [M, D]
  num_erase_units: int = 0


@dataclasses.dataclass
class ShrinkStats:
  """Sizes before and after shrinking."""
  d_model_before: int
  d_model_after: int
  dims_kept: int
  dims_reused: int
  heads_before: int
  heads_after: int
  mlp_units_before: int
  mlp_units_after: int
  erase_units: int
  params_before: int
  params_after: int
  trim_iterations: int


@dataclasses.dataclass
class ShrunkModel:
  """A model with per-layer shapes that matches an assembled model."""
  token_embed: np.ndarray  # [vocab, D]
  pos_embed: np.ndarray  # [T, D]
  layers: List[ShrunkLayer]
  unembed_matrix: np.ndarray  # [D, D_out]
  logit_divisor: np.float32
  categorical_output: bool
  causal: bool
  pad_token: Optional[int]
  slot_labels: List[List[str]]  # residual labels held by each slot, in order
  stats: Optional[ShrinkStats] = None
  input_encoder: Optional[encoder.Encoder] = None
  output_encoder: Optional[encoder.Encoder] = None

  @property
  def num_params(self) -> int:
    arrays = [self.token_embed, self.pos_embed]
    for layer in self.layers:
      arrays += [layer.wq, layer.wk, layer.wv, layer.wo, layer.w1, layer.w2]
    return int(sum(a.size for a in arrays))

  def _params(self) -> Dict[str, Any]:
    layers = [
        dict(wq=l.wq, wk=l.wk, wv=l.wv, wo=l.wo, w1=l.w1, w2=l.w2)
        for l in self.layers
    ]
    return dict(te=self.token_embed, pe=self.pos_embed, layers=layers,
                unembed=self.unembed_matrix)

  def output_values(self, tokens: jax.Array) -> jax.Array:
    """Residual projected onto the output space, [B, T, D_out]."""
    shapes = tuple((l.qk_sizes, l.v_sizes) for l in self.layers)
    return _forward(self._params(), jnp.asarray(tokens), shapes=shapes,
                    divisor=float(self.logit_divisor), causal=self.causal,
                    pad_token=self.pad_token)

  def apply(self, tokens: List[Any]) -> List[Any]:
    """Decoded outputs for one input sequence, like the assembled model."""
    ids = self.input_encoder.encode(tokens) if self.input_encoder else tokens
    out = self.output_values(jnp.array([ids]))[0]
    if self.categorical_output:
      out = jnp.argmax(out, axis=-1)
    elif out.shape[-1] == 1:
      out = out.squeeze(-1)
    decoded = out.tolist()
    if self.output_encoder:
      decoded = self.output_encoder.decode(decoded)
    if self.input_encoder is not None and self.input_encoder.bos_token:
      decoded = [self.input_encoder.bos_token] + decoded[1:]
    return decoded


@functools.partial(
    jax.jit, static_argnames=("shapes", "divisor", "causal", "pad_token"))
def _forward(params, tokens, *, shapes, divisor, causal, pad_token):
  """Same computation as model.CompiledTransformerModel, per-layer shapes."""
  seq_len = tokens.shape[-1]
  x = params["te"][tokens] + params["pe"][jnp.arange(seq_len)][None]
  if pad_token is None:
    mask = jnp.ones(tokens.shape, dtype=bool)
  else:
    mask = tokens != pad_token
  mask = jnp.repeat(mask[:, None, :], seq_len, axis=1)  # [B, T, T]
  if causal:
    mask = mask & jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None]
  div = jnp.asarray(divisor, dtype=x.dtype)
  for p, (qk_sizes, v_sizes) in zip(params["layers"], shapes):
    if v_sizes:
      q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]
      qk_edges = np.cumsum((0,) + qk_sizes)
      v_edges = np.cumsum((0,) + v_sizes)
      outs = []
      for h in range(len(v_sizes)):
        qs = slice(qk_edges[h], qk_edges[h + 1])
        logits = jnp.einsum("btd,bTd->btT", q[..., qs], k[..., qs]) / div
        logits = jnp.where(mask, logits, _MASKED_LOGIT)
        weights = jax.nn.softmax(logits)
        vs = slice(v_edges[h], v_edges[h + 1])
        outs.append(jnp.einsum("btT,bTd->btd", weights, v[..., vs]))
      x = x + jnp.concatenate(outs, axis=-1) @ p["wo"]
    if p["w1"].shape[1]:
      x = x + jax.nn.relu(x @ p["w1"]) @ p["w2"]
  return x @ params["unembed"]


@dataclasses.dataclass
class Comparison:
  """Agreement between an assembled model and its shrunk version."""
  sequences: int
  # Sequences with a differing output where the original is decided: a
  # categorical position whose top-2 score margin is at least tie_tol, or any
  # numerical position whose value is not bitwise equal.
  decoded_mismatches: int
  # Sequences whose only differences are at categorical ties of the original
  # (top-2 margin below tie_tol), where its argmax is set by rounding.
  tie_flips: int
  max_abs_diff: float  # largest output-space difference at a real position


def compare(model: assemble.AssembledTransformerModel, shrunk: ShrunkModel,
            inputs: List[List[Any]], batch_size: int,
            tie_tol: float) -> Comparison:
  """Runs both models on inputs (without BOS) and compares their outputs.

  Inputs are padded with PAD to a common length. Position 0 (BOS) and PAD
  positions are not compared, since the decoder overrides or ignores them.
  """
  enc = model.input_encoder
  bos, pad_id = enc.bos_token, enc.pad_encoding
  original = jax.jit(model.forward)
  unembed = jnp.asarray(model.unembed_matrix)
  mismatches, tie_flips, max_diff = 0, 0, 0.0
  for start in range(0, len(inputs), batch_size):
    chunk = inputs[start:start + batch_size]
    width = 1 + max(len(x) for x in chunk)
    ids = np.full((len(chunk), width), pad_id, dtype=np.int32)
    for r, x in enumerate(chunk):
      encoded = enc.encode([bos] + list(x))
      ids[r, :len(encoded)] = encoded
    real = ids != pad_id
    real[:, 0] = False
    a = np.asarray(original(model.params, jnp.asarray(ids))
                   .transformer_output.output @ unembed)
    b = np.asarray(shrunk.output_values(ids))
    max_diff = max(max_diff, float(np.abs(a - b)[real].max(initial=0.0)))
    if shrunk.categorical_output:
      differs = (a.argmax(-1) != b.argmax(-1)) & real
      top2 = np.sort(a, axis=-1)[..., -2:]
      tied = (top2[..., 1] - top2[..., 0]) < tie_tol
    else:
      differs = (a != b).any(-1) & real
      tied = np.zeros_like(differs)
    decided = (differs & ~tied).any(1)
    mismatches += int(decided.sum())
    tie_flips += int(((differs & tied).any(1) & ~decided).sum())
  return Comparison(sequences=len(inputs), decoded_mismatches=mismatches,
                    tie_flips=tie_flips, max_abs_diff=max_diff)


def _extract(model: assemble.AssembledTransformerModel):
  """Per-layer dense weights from the haiku params, as numpy arrays."""
  cfg = model.model_config
  d, nh, ks = len(model.residual_labels), cfg.num_heads, cfg.key_size
  params = {k: {n: np.asarray(w) for n, w in v.items()}
            for k, v in model.params.items()}
  for name, mod in params.items():
    if "b" in mod and np.any(mod["b"] != 0):
      raise ValueError(f"Shrinking assumes zero biases, but {name} has some.")
  layers = []
  for l in range(cfg.num_layers):
    pre = f"transformer/layer_{l}"
    layers.append(dict(
        q=params[f"{pre}/attn/query"]["w"].reshape(d, nh, ks),
        k=params[f"{pre}/attn/key"]["w"].reshape(d, nh, ks),
        v=params[f"{pre}/attn/value"]["w"].reshape(d, nh, ks),
        o=params[f"{pre}/attn/linear"]["w"].reshape(nh, ks, d),
        w1=params[f"{pre}/mlp/linear_1"]["w"],
        w2=params[f"{pre}/mlp/linear_2"]["w"],
    ))
  return params, layers


def _trim(layers, te, pe, unembed):
  """Fixed point of keep masks for dims, qk/v columns and MLP units."""
  d = te.shape[1]
  dims = np.ones(d, dtype=bool)
  qk = [np.ones(l["q"].shape[1:], dtype=bool) for l in layers]  # [H, K]
  vv = [np.ones(l["v"].shape[1:], dtype=bool) for l in layers]  # [H, K]
  units = [np.ones(l["w1"].shape[1], dtype=bool) for l in layers]
  out_read = (unembed != 0).any(1)
  emb_write = (te != 0).any(0) | (pe != 0).any(0)
  iterations = 0
  while True:
    iterations += 1
    read, written = out_read.copy(), emb_write.copy()
    new_qk, new_vv, new_units = [], [], []
    for l, layer in enumerate(layers):
      nzq = (layer["q"][dims] != 0).any(0)  # [H, K]
      nzk = (layer["k"][dims] != 0).any(0)
      nzv = (layer["v"][dims] != 0).any(0)
      nzo = (layer["o"][:, :, dims] != 0).any(-1)
      v_keep = vv[l] & nzv & nzo
      head_alive = v_keep.any(1, keepdims=True)
      qk_keep = qk[l] & nzq & nzk & head_alive
      unit_keep = (units[l] & (layer["w1"][dims] != 0).any(0)
                   & (layer["w2"][:, dims] != 0).any(1))
      read |= (layer["q"][:, qk_keep] != 0).any(1)
      read |= (layer["k"][:, qk_keep] != 0).any(1)
      read |= (layer["v"][:, v_keep] != 0).any(1)
      read |= (layer["w1"][:, unit_keep] != 0).any(1)
      written |= (layer["o"][v_keep] != 0).any(0)
      written |= (layer["w2"][unit_keep] != 0).any(0)
      new_qk.append(qk_keep)
      new_vv.append(v_keep)
      new_units.append(unit_keep)
    new_dims = dims & read & written
    changed = (not np.array_equal(new_dims, dims)
               or any(not np.array_equal(a, b) for a, b in zip(new_qk, qk))
               or any(not np.array_equal(a, b) for a, b in zip(new_vv, vv))
               or any(not np.array_equal(a, b)
                      for a, b in zip(new_units, units)))
    dims, qk, vv, units = new_dims, new_qk, new_vv, new_units
    if not changed:
      return dims, qk, vv, units, iterations


def _lifetimes(layers, te, pe, unembed, qk, vv, units):
  """Birth (first write), last write and death (last read) per kept dim.

  Time 0 is the embedding, 2l+1 the attention and 2l+2 the MLP of layer l,
  and 2L+1 the unembedding.
  """
  d = te.shape[1]
  end = 2 * len(layers) + 1
  birth = np.full(d, np.inf)
  last_write = np.full(d, -np.inf)
  death = np.full(d, -np.inf)
  emb = (te != 0).any(0) | (pe != 0).any(0)
  birth[emb], last_write[emb] = 0, 0
  for l, layer in enumerate(layers):
    t_attn, t_mlp = 2 * l + 1, 2 * l + 2
    reads_attn = ((layer["q"][:, qk[l]] != 0).any(1)
                  | (layer["k"][:, qk[l]] != 0).any(1)
                  | (layer["v"][:, vv[l]] != 0).any(1))
    reads_mlp = (layer["w1"][:, units[l]] != 0).any(1)
    writes_attn = (layer["o"][vv[l]] != 0).any(0)
    writes_mlp = (layer["w2"][units[l]] != 0).any(0)
    for t, rd, wr in ((t_attn, reads_attn, writes_attn),
                      (t_mlp, reads_mlp, writes_mlp)):
      death[rd] = np.maximum(death[rd], t)
      birth[wr] = np.minimum(birth[wr], t)
      last_write[wr] = np.maximum(last_write[wr], t)
  death[(unembed != 0).any(1)] = end
  return birth, last_write, death, end


def _assign_slots(dims, birth, last_write, death, end, reuse):
  """Map kept dims to slots; returns slot per dim and erase time per dim."""
  kept = np.flatnonzero(dims)
  slot_of = np.full(len(dims), -1)
  erase_at = np.full(len(dims), -1)
  if not reuse:
    slot_of[kept] = np.arange(len(kept))
    return slot_of, erase_at, len(kept)
  # Erase in the MLP of the last reading layer: same MLP if the MLP read it.
  erase_time = np.where(death % 2 == 0, death, death + 1)
  erasable = (death < end) & (last_write < erase_time)
  free_after = []  # per slot: time its current value can be erased, or inf
  occupant = []
  for i in kept[np.argsort(birth[kept], kind="stable")]:
    choice = -1
    for s, t_free in enumerate(free_after):
      if t_free < birth[i]:
        choice = s
        break
    if choice < 0:
      free_after.append(np.inf)
      occupant.append(-1)
      choice = len(free_after) - 1
    else:
      erase_at[occupant[choice]] = erase_time[occupant[choice]]
    slot_of[i] = choice
    occupant[choice] = i
    free_after[choice] = erase_time[i] if erasable[i] else np.inf
  return slot_of, erase_at, len(free_after)


def _place_rows(target, slot_of, src, dims_mask):
  """Copy nonzero rows src[i] for dims in dims_mask into target[slot_of[i]].

  A nonzero row means the dim is read or written at this time, so it is live
  and owns its slot. Zero rows of dead dims that share the slot are skipped.
  """
  idx = np.flatnonzero(dims_mask & (src != 0).any(1))
  slots = slot_of[idx]
  if len(np.unique(slots)) != len(slots):
    raise AssertionError("Two live dims share a slot in one read or write.")
  target[slots] = src[idx]


def shrink(model: assemble.AssembledTransformerModel,
           reuse: bool = False) -> ShrunkModel:
  """Returns a smaller model with the same outputs as `model`.

  Args:
    model: an assembled model from compiling.compile_rasp_to_model.
    reuse: if True, residual dims are reused after their value is dead. This
      narrows the residual stream, but adds two MLP units per erased dim.
  """
  if model.unembed_matrix is None:
    raise ValueError("Model has no unembed_matrix; recompile it.")
  params, layers = _extract(model)
  te = params["token_embed"]["embeddings"]
  pe = params["pos_embed"]["embeddings"]
  unembed = np.asarray(model.unembed_matrix, dtype=te.dtype)
  dims, qk, vv, units, iterations = _trim(layers, te, pe, unembed)
  birth, last_write, death, end = _lifetimes(layers, te, pe, unembed, qk, vv,
                                             units)
  slot_of, erase_at, n_slots = _assign_slots(dims, birth, last_write, death,
                                             end, reuse)
  dtype = te.dtype

  def readable(t):
    # Dims with a value at time t: kept and born before t. Rows for other dims
    # would read zeros in the original model, so dropping them is exact.
    return dims & (birth < t)

  def writable(t):
    return dims & (birth <= t) & (last_write >= t)

  new_te = np.zeros((te.shape[0], n_slots), dtype)
  new_pe = np.zeros((pe.shape[0], n_slots), dtype)
  _place_rows(new_te.T, slot_of, te.T, writable(0))
  _place_rows(new_pe.T, slot_of, pe.T, writable(0))
  new_unembed = np.zeros((n_slots, unembed.shape[1]), dtype)
  _place_rows(new_unembed, slot_of, unembed, readable(end))

  new_layers = []
  heads_after = 0
  units_after = 0
  for l, layer in enumerate(layers):
    t_attn, t_mlp = 2 * l + 1, 2 * l + 2
    wq_cols, wk_cols, wv_cols, wo_rows, qk_sizes, v_sizes = [], [], [], [], [], []
    for h in range(layer["q"].shape[1]):
      if not vv[l][h].any():
        continue
      qk_sizes.append(int(qk[l][h].sum()))
      v_sizes.append(int(vv[l][h].sum()))
      wq_cols.append(layer["q"][:, h, qk[l][h]])
      wk_cols.append(layer["k"][:, h, qk[l][h]])
      wv_cols.append(layer["v"][:, h, vv[l][h]])
      wo_rows.append(layer["o"][h, vv[l][h]])
    heads_after += len(v_sizes)
    rd, wr = readable(t_attn), writable(t_attn)

    def gather_rows(cols, width):
      out = np.zeros((n_slots, width), dtype)
      if width:
        _place_rows(out, slot_of, np.concatenate(cols, axis=1), rd)
      return out

    wq = gather_rows(wq_cols, sum(qk_sizes))
    wk = gather_rows(wk_cols, sum(qk_sizes))
    wv = gather_rows(wv_cols, sum(v_sizes))
    wo = np.zeros((sum(v_sizes), n_slots), dtype)
    if v_sizes:
      _place_rows(wo.T, slot_of, np.concatenate(wo_rows, axis=0).T, wr)

    rd, wr = readable(t_mlp), writable(t_mlp)
    kept_units = np.flatnonzero(units[l])
    erased = np.flatnonzero(erase_at == t_mlp)
    n_units = len(kept_units) + 2 * len(erased)
    w1 = np.zeros((n_slots, n_units), dtype)
    w2 = np.zeros((n_units, n_slots), dtype)
    n_kept = len(kept_units)
    _place_rows(w1[:, :n_kept], slot_of, layer["w1"][:, kept_units], rd)
    _place_rows(w2[:n_kept].T, slot_of, layer["w2"][kept_units].T, wr)
    for j, i in enumerate(erased):
      pos, neg = len(kept_units) + 2 * j, len(kept_units) + 2 * j + 1
      s = slot_of[i]
      if w2[:, s].any():
        raise AssertionError("A dim is written in the MLP that erases it.")
      w1[s, pos], w2[pos, s] = 1, -1  # -ReLU(x)
      w1[s, neg], w2[neg, s] = -1, 1  # +ReLU(-x)
    units_after += n_units
    new_layers.append(ShrunkLayer(
        wq=wq, wk=wk, wv=wv, wo=wo, qk_sizes=tuple(qk_sizes),
        v_sizes=tuple(v_sizes), w1=w1, w2=w2,
        num_erase_units=2 * len(erased)))

  slot_labels = [[] for _ in range(n_slots)]
  for i in np.flatnonzero(dims)[np.argsort(birth[dims], kind="stable")]:
    slot_labels[slot_of[i]].append(model.residual_labels[i])

  cfg = model.model_config
  shrunk = ShrunkModel(
      token_embed=new_te, pos_embed=new_pe, layers=new_layers,
      unembed_matrix=new_unembed,
      logit_divisor=np.sqrt(cfg.key_size).astype(dtype),
      categorical_output=model.categorical_output, causal=cfg.causal,
      pad_token=model.pad_token, slot_labels=slot_labels,
      input_encoder=model.input_encoder, output_encoder=model.output_encoder)
  shrunk.stats = ShrinkStats(
      d_model_before=len(model.residual_labels), d_model_after=n_slots,
      dims_kept=int(dims.sum()), dims_reused=int(dims.sum()) - n_slots,
      heads_before=cfg.num_layers * cfg.num_heads, heads_after=heads_after,
      mlp_units_before=cfg.num_layers * cfg.mlp_hidden_size,
      mlp_units_after=units_after,
      erase_units=int(sum(l.num_erase_units for l in new_layers)),
      params_before=int(sum(np.size(w) for mod in params.values()
                            for w in mod.values())),
      params_after=shrunk.num_params, trim_iterations=iterations)
  return shrunk

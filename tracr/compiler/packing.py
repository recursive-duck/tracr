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
"""Pack several compiled programs into one model, selected by the input.

Each program becomes a task. Token ids carry the task: task i owns ids
[offset_i, offset_i + vocab_i). Positions use one shared table. For every
task, the packed model gives the same outputs as that program alone.

Only one task runs at a time, so programs can share residual dims. A block of
another program must then write exactly zero:

* Attention heads: if every dim a head reads into its values is zero, the head
  writes zero, whatever its attention pattern. So a value read of a dim at time
  t must not meet a value of another program alive in the same slot at t.
* MLP units: each unit gets weight -M on the `one` dim of every other program.
  That dim is 1 at every position of its own task and 0 otherwise, so the gate
  adds an exact 0 for the unit's own task and pushes the unit below zero for
  any other. M for task j is twice a sound upper bound on the unit's input
  while task j runs, plus one: only task j's values born before the unit's
  layer and the shared positions are nonzero then. Bounds are intervals
  pushed through the layers; categorical dims also stay in [0, 1].
  Each unit's input and output weights are then balanced by a power of 2,
  which keeps outputs bitwise equal and the largest weights smaller.

Values of one program never share a slot (nothing is erased). Two values of
different programs conflict if one is read into attention values after the
other is born. Slots come from a DSatur colouring of this conflict graph. A
clique bound certifies it; if they differ, an integer program searches for a
colouring with fewer slots.

Input dims are special: each program's `one` dim gets its own slot (it gates
the others), and positional dims are shared by all tasks, so they also get
their own slots. All programs start at layer 0.
"""

import dataclasses
import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize
from scipy import sparse

from tracr.compiler import assemble
from tracr.compiler import shrinking
from tracr.transformer import encoder

_MASKED_LOGIT = -1e30
_ONE_LABEL = "one"
# Default gate: M = GATE_SCALE * bound + GATE_OFFSET for a unit's input bound.
GATE_SCALE = 2.0
GATE_OFFSET = 1.0
# Bound categorical dims by [0, 1] when sizing gates.
ONEHOT_BOUNDS = True
# Balance each unit's input and output weights with a power-of-2 scale.
REBALANCE = True
# Default time limit for the integer program, in seconds.
ILP_TIME_LIMIT_S = 60.0
# Skip the integer program if its constraint matrix has more nonzeros.
ILP_MAX_NONZEROS = 20_000_000


@dataclasses.dataclass
class PackedLayer:
  """One layer: heads and MLP units of every program side by side."""
  wq: np.ndarray  # [D, sum(qk_sizes)]
  wk: np.ndarray
  wv: np.ndarray  # [D, sum(v_sizes)]
  wo: np.ndarray  # [sum(v_sizes), D]
  qk_sizes: Tuple[int, ...]
  v_sizes: Tuple[int, ...]
  divisors: Tuple[float, ...]  # logit divisor per head, from its program
  w1: np.ndarray  # [D, units]
  w2: np.ndarray  # [units, D]
  unit_task: np.ndarray  # [units] task that owns each MLP unit


@dataclasses.dataclass
class PackedTask:
  """How to feed one program's inputs and read its outputs."""
  name: str
  token_offset: int
  vocab_size: int
  unembed_matrix: np.ndarray  # [D, D_out]
  categorical_output: bool
  input_encoder: Optional[encoder.Encoder] = None
  output_encoder: Optional[encoder.Encoder] = None


@dataclasses.dataclass
class PackStats:
  """Sizes and layout quality."""
  programs: int
  d_model: int
  d_programs: List[int]  # residual width each program needs on its own
  gate_slots: int
  position_slots: int
  slots_lower_bound: int  # for the shared slots, from a clique
  slots_dsatur: int
  slots_final: int
  solver: str  # "dsatur" or "ilp"
  ilp_status: str
  heads: int
  mlp_units: int
  params: int
  params_block_diagonal: int  # same blocks, no shared dims
  max_gate: float  # largest gate before rebalancing
  max_weight: float  # largest |weight| of the packed model


@dataclasses.dataclass
class PackedModel:
  """One model; the token ids choose which program runs."""
  token_embed: np.ndarray  # [sum vocab, D]
  pos_embed: np.ndarray  # [T, D]
  is_pad: np.ndarray  # [sum vocab] bool
  layers: List[PackedLayer]
  tasks: List[PackedTask]
  causal: bool
  slot_labels: List[List[str]]  # "task/label" held by each slot
  stats: Optional[PackStats] = None

  @property
  def num_params(self) -> int:
    arrays = [self.token_embed, self.pos_embed]
    for layer in self.layers:
      arrays += [layer.wq, layer.wk, layer.wv, layer.wo, layer.w1, layer.w2]
    return int(sum(a.size for a in arrays))

  def residual(self, packed_ids: jax.Array) -> jax.Array:
    """Final residual stream for packed token ids, [B, T, D]."""
    shapes = tuple((l.qk_sizes, l.v_sizes, l.divisors) for l in self.layers)
    layers = [dict(wq=l.wq, wk=l.wk, wv=l.wv, wo=l.wo, w1=l.w1, w2=l.w2)
              for l in self.layers]
    params = dict(te=self.token_embed, pe=self.pos_embed, is_pad=self.is_pad,
                  layers=layers)
    return _forward(params, jnp.asarray(packed_ids), shapes=shapes,
                    causal=self.causal)

  def output_values(self, task: int, ids: Any) -> jax.Array:
    """Outputs of `task` for its own (unpacked) token ids, [B, T, D_out]."""
    t = self.tasks[task]
    packed = jnp.asarray(ids) + t.token_offset
    return self.residual(packed) @ jnp.asarray(t.unembed_matrix)

  def task(self, task: int) -> "TaskView":
    return TaskView(self, task)


@dataclasses.dataclass
class TaskView:
  """One task of a packed model, usable with shrinking.compare."""
  packed: PackedModel
  index: int

  @property
  def categorical_output(self) -> bool:
    return self.packed.tasks[self.index].categorical_output

  def output_values(self, ids: Any) -> jax.Array:
    return self.packed.output_values(self.index, ids)


@functools.partial(jax.jit, static_argnames=("shapes", "causal"))
def _forward(params, tokens, *, shapes, causal):
  seq_len = tokens.shape[-1]
  x = params["te"][tokens] + params["pe"][jnp.arange(seq_len)][None]
  mask = ~params["is_pad"][tokens]
  mask = jnp.repeat(mask[:, None, :], seq_len, axis=1)
  if causal:
    mask = mask & jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[None]
  for p, (qk_sizes, v_sizes, divisors) in zip(params["layers"], shapes):
    if v_sizes:
      q, k, v = x @ p["wq"], x @ p["wk"], x @ p["wv"]
      qk_edges = np.cumsum((0,) + qk_sizes)
      v_edges = np.cumsum((0,) + v_sizes)
      outs = []
      for h in range(len(v_sizes)):
        qs = slice(qk_edges[h], qk_edges[h + 1])
        div = jnp.asarray(divisors[h], dtype=x.dtype)
        logits = jnp.einsum("btd,bTd->btT", q[..., qs], k[..., qs]) / div
        logits = jnp.where(mask, logits, _MASKED_LOGIT)
        weights = jax.nn.softmax(logits)
        vs = slice(v_edges[h], v_edges[h + 1])
        outs.append(jnp.einsum("btT,bTd->btd", weights, v[..., vs]))
      x = x + jnp.concatenate(outs, axis=-1) @ p["wo"]
    if p["w1"].shape[1]:
      x = x + jax.nn.relu(x @ p["w1"]) @ p["w2"]
  return x


@dataclasses.dataclass
class _Program:
  """One program after trimming, in its own residual coordinates."""
  name: str
  model: assemble.AssembledTransformerModel
  te: np.ndarray
  pe: np.ndarray
  unembed: np.ndarray
  layers: List[Dict[str, np.ndarray]]
  dims: np.ndarray  # kept dims
  qk: List[np.ndarray]
  vv: List[np.ndarray]
  units: List[np.ndarray]
  birth: np.ndarray
  last_write: np.ndarray
  one: int  # index of the `one` dim
  pos: np.ndarray  # dims written by the positional embedding
  divisor: float

  def readable(self, t):
    return self.dims & (self.birth < t)

  def writable(self, t):
    return self.dims & (self.birth <= t) & (self.last_write >= t)


def _load(name, model) -> _Program:
  if model.unembed_matrix is None:
    raise ValueError(f"{name}: model has no unembed_matrix; recompile it.")
  params, layers = shrinking._extract(model)  # pylint: disable=protected-access
  te = params["token_embed"]["embeddings"]
  pe = params["pos_embed"]["embeddings"]
  unembed = np.asarray(model.unembed_matrix, dtype=te.dtype)
  if _ONE_LABEL not in model.residual_labels:
    raise ValueError(f"{name}: no '{_ONE_LABEL}' dim to gate other tasks with.")
  one = model.residual_labels.index(_ONE_LABEL)
  if not np.all(te[:, one] == 1) or np.any(pe[:, one] != 0):
    raise ValueError(f"{name}: '{_ONE_LABEL}' is not 1 on every token.")
  pos = (pe != 0).any(0)
  if np.any(pos & (te != 0).any(0)):
    raise ValueError(f"{name}: a dim is written by both embeddings.")
  dims, qk, vv, units, _ = shrinking._trim(layers, te, pe, unembed)  # pylint: disable=protected-access
  birth, last_write, _, _ = shrinking._lifetimes(  # pylint: disable=protected-access
      layers, te, pe, unembed, qk, vv, units)
  inputs = pos.copy()
  inputs[one] = True
  if np.any(dims & inputs & (last_write > 0)):
    raise ValueError(f"{name}: a layer writes into an input dim.")
  return _Program(
      name=name, model=model, te=te, pe=pe, unembed=unembed, layers=layers,
      dims=dims, qk=qk, vv=vv, units=units, birth=birth,
      last_write=last_write, one=one, pos=pos,
      divisor=float(np.sqrt(model.model_config.key_size)))


def _value_reads(p: _Program) -> np.ndarray:
  """Last time each dim is read into attention values, -1 if never."""
  last = np.full(len(p.dims), -1)
  for l, layer in enumerate(p.layers):
    t = 2 * l + 1
    rows = (layer["v"][:, p.vv[l]] != 0).any(1) & p.readable(t)
    last[rows] = t
  return last


def _affine(w, lo, hi):
  """Interval of w.T @ x for x in the box [lo, hi]."""
  w = w.astype(np.float64)
  pos, neg = np.maximum(w, 0), np.minimum(w, 0)
  return pos.T @ lo + neg.T @ hi, pos.T @ hi + neg.T @ lo


def _intervals(p: _Program, onehot: bool) -> Tuple[np.ndarray, np.ndarray]:
  """Sound [lo, hi] of each dim after each time, as two [2L + 1, d] arrays.

  Row t holds the residual after time t (0 = embedding), so a block at time t
  reads row t - 1. With onehot, categorical dims (labels "name:value") are also
  kept in [0, 1]: tracr writes one-hot vectors or averages of them there.
  """
  d = len(p.dims)
  cat = np.array([":" in s for s in p.model.residual_labels]) & onehot

  def clamp(lo, hi):
    return (np.where(cat, np.clip(lo, 0, 1), lo),
            np.where(cat, np.clip(hi, 0, 1), hi))

  lo, hi = clamp(p.te.min(0) + p.pe.min(0), p.te.max(0) + p.pe.max(0))
  lo, hi = lo.astype(np.float64), hi.astype(np.float64)
  rows_lo, rows_hi = [lo], [hi]
  for l, layer in enumerate(p.layers):
    t_attn, t_mlp = 2 * l + 1, 2 * l + 2
    rd = p.readable(t_attn)
    w_lo, w_hi = np.zeros(d), np.zeros(d)
    for h in range(layer["v"].shape[1]):
      cols = p.vv[l][h]
      if not cols.any():
        continue
      # Attention averages value vectors, so each column stays in its box.
      v_lo, v_hi = _affine(layer["v"][rd][:, h, cols], lo[rd], hi[rd])
      o_lo, o_hi = _affine(layer["o"][h, cols], v_lo, v_hi)
      w_lo, w_hi = w_lo + o_lo, w_hi + o_hi
    wr = p.writable(t_attn)
    lo, hi = clamp(lo + w_lo * wr, hi + w_hi * wr)
    rows_lo.append(lo)
    rows_hi.append(hi)
    rd, units = p.readable(t_mlp), p.units[l]
    h_lo, h_hi = _affine(layer["w1"][rd][:, units], lo[rd], hi[rd])
    o_lo, o_hi = _affine(layer["w2"][units], np.maximum(h_lo, 0),
                         np.maximum(h_hi, 0))
    wr = p.writable(t_mlp)
    lo, hi = clamp(lo + o_lo * wr, hi + o_hi * wr)
    rows_lo.append(lo)
    rows_hi.append(hi)
  return np.stack(rows_lo), np.stack(rows_hi)


def _conflicts(prog_of, birth, vread):
  """Adjacency: same program, or a value read of one after the other's birth."""
  same = prog_of[:, None] == prog_of[None, :]
  cross = (vread[None, :] > birth[:, None]) | (vread[:, None] > birth[None, :])
  adj = same | cross
  np.fill_diagonal(adj, False)
  return adj


def _dsatur(adj: np.ndarray) -> np.ndarray:
  """Greedy colouring: most distinct neighbour colours first, then degree."""
  n = len(adj)
  color = np.full(n, -1)
  seen = np.zeros((n, n + 1), dtype=bool)  # [vertex, colour used by neighbour]
  sat = np.zeros(n, dtype=int)
  degree = adj.sum(1)
  for _ in range(n):
    key = np.where(color < 0, sat * (n + 1) + degree, -1)
    v = int(np.argmax(key))
    c = int(np.argmin(seen[v]))  # smallest colour no neighbour uses
    color[v] = c
    nb = adj[v] & ~seen[:, c]
    seen[nb, c] = True
    sat[nb] += 1
  return color


def _cliques(prog_of, birth, vread, num_programs, end):
  """Cliques covering every conflict edge, one per (program, time).

  At time t, the values of program P born before t, plus the values of other
  programs born before t and read into attention values at or after t, are
  pairwise in conflict.
  """
  times = sorted(set(int(t) for t in vread if t >= 0) | {end})
  cliques = []
  for t in times:
    later = (birth < t) & (vread >= t)
    for p in range(num_programs):
      members = ((prog_of == p) & (birth < t)) | (later & (prog_of != p))
      if members.any():
        cliques.append(np.flatnonzero(members))
  return cliques


def _ilp(n, cliques, fixed, s_max, time_limit_s, max_nonzeros):
  """Colouring with the fewest slots, at most s_max; (None, status) if none.

  x[v, s] = 1 puts value v in slot s; y[s] = 1 marks slot s used. Every value
  takes one slot, every clique has at most one member per used slot, and used
  slots come first. Minimises the number of used slots.
  """
  nx_, ny = n * s_max, s_max
  sizes = np.array([len(c) for c in cliques])
  nnz = (n + sizes.sum() + len(cliques)) * s_max + 2 * (s_max - 1)
  if nnz > max_nonzeros:
    return None, f"skipped: {nnz:,} nonzeros > {max_nonzeros:,}"
  slots = np.arange(s_max)
  # Every value gets one slot: rows 0..n-1.
  r1 = np.repeat(np.arange(n), s_max)
  c1 = np.arange(nx_)
  # Clique c, slot s: sum of members' x[v, s] - y[s] <= 0.
  members = np.concatenate(cliques)
  cid = np.repeat(np.arange(len(cliques)), sizes)
  r2 = n + (cid[:, None] * s_max + slots).ravel()
  c2 = (members[:, None] * s_max + slots).ravel()
  r3 = n + np.arange(len(cliques) * s_max)
  c3 = nx_ + np.tile(slots, len(cliques))
  # y[s] >= y[s + 1].
  base = n + len(cliques) * s_max
  r4 = base + np.repeat(np.arange(s_max - 1), 2)
  c4 = nx_ + np.stack([slots[:-1], slots[1:]], 1).ravel()
  v4 = np.tile([1.0, -1.0], s_max - 1)
  rows = np.concatenate([r1, r2, r3, r4])
  cols = np.concatenate([c1, c2, c3, c4])
  vals = np.concatenate([np.ones(len(c1) + len(c2)), -np.ones(len(c3)), v4])
  num_rows = base + s_max - 1
  lo = np.concatenate([np.ones(n), np.full(base - n, -np.inf),
                       np.zeros(s_max - 1)])
  hi = np.concatenate([np.ones(n), np.zeros(base - n),
                       np.full(s_max - 1, np.inf)])
  lb_x = np.zeros(nx_ + ny)
  ub_x = np.ones(nx_ + ny)
  # Symmetry: one clique's members take slots 0..k-1.
  lb_x[np.asarray(fixed) * s_max + np.arange(len(fixed))] = 1
  lb_x[nx_ + np.arange(len(fixed))] = 1
  a = sparse.csr_array((vals, (rows, cols)), shape=(num_rows, nx_ + ny))
  cost = np.concatenate([np.zeros(nx_), np.ones(ny)])
  res = optimize.milp(
      cost, constraints=optimize.LinearConstraint(a, lo, hi),
      integrality=np.ones(nx_ + ny), bounds=optimize.Bounds(lb_x, ub_x),
      options=dict(time_limit=time_limit_s))
  status = f"{res.status}: {res.message}"
  if res.x is None:
    return None, status
  x = np.round(res.x[:nx_]).reshape(n, s_max)
  _, color = np.unique(x.argmax(1), return_inverse=True)
  return color, status


def pack(models: Sequence[assemble.AssembledTransformerModel],
         names: Optional[Sequence[str]] = None,
         gate_scale: float = GATE_SCALE,
         gate_offset: float = GATE_OFFSET,
         ilp_time_limit_s: float = ILP_TIME_LIMIT_S,
         ilp_max_nonzeros: int = ILP_MAX_NONZEROS,
         force_ilp: bool = False,
         onehot_bounds: bool = ONEHOT_BOUNDS,
         rebalance: bool = REBALANCE) -> PackedModel:
  """Packs compiled programs into one model; task i runs models[i].

  Args:
    models: assembled models with the same max_seq_len and causal setting.
    names: task names, default "task0", "task1", ...
    gate_scale, gate_offset: a unit's gate for task j is gate_scale times a
      bound on its input while task j runs, plus gate_offset.
    ilp_time_limit_s: time limit for the integer program.
    ilp_max_nonzeros: skip the integer program if its matrix is larger.
    force_ilp: run the integer program even if DSatur meets the clique bound.
    onehot_bounds: bound categorical dims by [0, 1] (tracr writes one-hot
      vectors or averages of them there), not only by interval arithmetic.
    rebalance: scale each unit's input weights by 1/c and output weights by c,
      c a power of 2, so its largest input and output weights are balanced.
      ReLU(x / c) * c = ReLU(x), and powers of 2 scale floats exactly.
  """
  names = list(names) if names is not None else [
      f"task{i}" for i in range(len(models))]
  progs = [_load(n, m) for n, m in zip(names, models)]
  if len({p.pe.shape[0] for p in progs}) != 1:
    raise ValueError("Programs must share max_seq_len.")
  if len({p.model.model_config.causal for p in progs}) != 1:
    raise ValueError("Programs must share the causal setting.")
  dtype = progs[0].te.dtype

  # Shared positional dims, keyed by label; their columns must agree.
  pos_cols: Dict[str, np.ndarray] = {}
  for p in progs:
    for i in np.flatnonzero(p.pos & p.dims):
      label = p.model.residual_labels[i]
      col = p.pe[:, i]
      if label in pos_cols and not np.array_equal(pos_cols[label], col):
        raise ValueError(f"Positional dim {label} differs between programs.")
      pos_cols[label] = col
  pos_labels = sorted(pos_cols)

  # Slots: gates, then positions, then the coloured values.
  k = len(progs)
  slot_of = [np.full(len(p.dims), -1) for p in progs]
  slot_labels: List[List[str]] = []
  for i, p in enumerate(progs):
    slot_of[i][p.one] = i
    slot_labels.append([f"{p.name}/{_ONE_LABEL}"])
  for j, label in enumerate(pos_labels):
    slot_labels.append([label])
    for i, p in enumerate(progs):
      if label in p.model.residual_labels:
        slot_of[i][p.model.residual_labels.index(label)] = k + j
  base = k + len(pos_labels)

  prog_of, birth, vread, owner = [], [], [], []
  for i, p in enumerate(progs):
    reads = _value_reads(p)
    if np.any(reads[p.pos & p.dims] >= 0):
      raise NotImplementedError(
          f"{p.name}: a head reads a positional dim into its values, so it "
          "would not be silent under other tasks.")
    for d in np.flatnonzero(p.dims & ~p.pos):
      if d == p.one:
        continue
      prog_of.append(i)
      birth.append(p.birth[d])
      vread.append(reads[d])
      owner.append((i, d))
  prog_of, birth, vread = (np.asarray(prog_of), np.asarray(birth),
                           np.asarray(vread))
  n = len(owner)
  end = 2 * max(len(p.layers) for p in progs) + 1
  adj = _conflicts(prog_of, birth, vread)
  color = _dsatur(adj) if n else np.zeros(0, dtype=int)
  dsatur_slots = int(color.max()) + 1 if n else 0
  cliques = _cliques(prog_of, birth, vread, k, end)
  biggest = max(cliques, key=len) if cliques else np.zeros(0, dtype=int)
  lower = len(biggest)
  solver, status = "dsatur", "not run: DSatur meets the clique bound"
  if n and (force_ilp or dsatur_slots > lower):
    ilp_color, status = _ilp(n, cliques, biggest, dsatur_slots,
                             ilp_time_limit_s, ilp_max_nonzeros)
    if ilp_color is not None and ilp_color.max() < color.max():
      color, solver = ilp_color, "ilp"
  if n and np.any(adj & (color[:, None] == color[None, :])):
    raise AssertionError("Two conflicting values share a slot.")
  shared = int(color.max()) + 1 if n else 0
  for v, (i, d) in enumerate(owner):
    slot_of[i][d] = base + color[v]
  slot_labels += [[] for _ in range(shared)]
  for v in np.argsort(birth, kind="stable"):
    i, d = owner[v]
    slot_labels[base + color[v]].append(
        f"{progs[i].name}/{progs[i].model.residual_labels[d]}")
  width = base + shared

  # Interval of each slot while task i runs, after each time: [k, end, width].
  # Only task i's values and the shared positions are nonzero then.
  slot_lo = np.zeros((k, end, width))
  slot_hi = np.zeros((k, end, width))
  for i, p in enumerate(progs):
    lo, hi = _intervals(p, onehot_bounds)
    rows = np.minimum(np.arange(end), len(lo) - 1)  # values persist
    live = slot_of[i] >= 0
    slot_lo[i][:, slot_of[i][live]] = lo[rows][:, live]
    slot_hi[i][:, slot_of[i][live]] = hi[rows][:, live]
    for j, label in enumerate(pos_labels):
      slot_lo[i, :, k + j] = pos_cols[label].min()
      slot_hi[i, :, k + j] = pos_cols[label].max()

  def place(target, i, src, mask):
    """Rows src[d] for dims in mask go to target[slot_of[d]]."""
    idx = np.flatnonzero(mask & (src != 0).any(1))
    slots = slot_of[i][idx]
    if np.any(slots < 0) or len(np.unique(slots)) != len(slots):
      raise AssertionError("A live dim has no slot or shares one.")
    target[slots] = src[idx]

  # Embeddings.
  offsets = np.cumsum([0] + [p.te.shape[0] for p in progs])
  te = np.zeros((offsets[-1], width), dtype)
  is_pad = np.zeros(offsets[-1], dtype=bool)
  pe = np.zeros((progs[0].pe.shape[0], width), dtype)
  for j, label in enumerate(pos_labels):
    pe[:, k + j] = pos_cols[label]
  tasks = []
  for i, p in enumerate(progs):
    rows = slice(offsets[i], offsets[i + 1])
    emb_dims = p.writable(0) & ~p.pos
    emb_dims[p.one] = True
    place(te[rows].T, i, p.te.T, emb_dims)
    pad = p.model.pad_token
    if pad is not None:
      is_pad[offsets[i] + pad] = True
    unembed = np.zeros((width, p.unembed.shape[1]), dtype)
    place(unembed, i, p.unembed, p.readable(end))
    tasks.append(PackedTask(
        name=p.name, token_offset=int(offsets[i]), vocab_size=p.te.shape[0],
        unembed_matrix=unembed, categorical_output=p.model.categorical_output,
        input_encoder=p.model.input_encoder,
        output_encoder=p.model.output_encoder))

  # Layers.
  layers = []
  max_gate = 0.0
  for l in range(max(len(p.layers) for p in progs)):
    t_attn, t_mlp = 2 * l + 1, 2 * l + 2
    wq, wk, wv, wo, w1, w2, unit_task = [], [], [], [], [], [], []
    qk_sizes, v_sizes, divisors = [], [], []
    for i, p in enumerate(progs):
      if l >= len(p.layers):
        continue
      layer = p.layers[l]
      rd, wr = p.readable(t_attn), p.writable(t_attn)
      for h in range(layer["q"].shape[1]):
        if not p.vv[l][h].any():
          continue
        qcols, vcols = p.qk[l][h], p.vv[l][h]
        for target, src, cols in ((wq, layer["q"][:, h], qcols),
                                  (wk, layer["k"][:, h], qcols),
                                  (wv, layer["v"][:, h], vcols)):
          block = np.zeros((width, int(cols.sum())), dtype)
          place(block, i, src[:, cols], rd)
          target.append(block)
        block = np.zeros((int(vcols.sum()), width), dtype)
        place(block.T, i, layer["o"][h, vcols].T, wr)
        wo.append(block)
        qk_sizes.append(int(qcols.sum()))
        v_sizes.append(int(vcols.sum()))
        divisors.append(p.divisor)
      units = np.flatnonzero(p.units[l])
      if len(units):
        rd, wr = p.readable(t_mlp), p.writable(t_mlp)
        b1 = np.zeros((width, len(units)), dtype)
        place(b1, i, layer["w1"][:, units], rd)
        b2 = np.zeros((len(units), width), dtype)
        place(b2.T, i, layer["w2"][units].T, wr)
        # Largest input of each unit while task j runs, before gating: [k, u].
        pos_w = np.maximum(b1, 0).astype(np.float64)
        neg_w = np.minimum(b1, 0).astype(np.float64)
        need = slot_hi[:, t_mlp - 1] @ pos_w + slot_lo[:, t_mlp - 1] @ neg_w
        gate = gate_scale * np.maximum(need, 0) + gate_offset
        others = [j for j in range(k) if j != i]
        b1[others] = -gate[others].astype(dtype)
        if others:
          max_gate = max(max_gate, float(gate[others].max()))
        if rebalance:
          a, b = np.abs(b1).max(0), np.abs(b2).max(1)
          ok = (a > 0) & (b > 0)
          shift = np.round(np.log2(np.where(ok, a, 1) / np.where(ok, b, 1)) / 2)
          c = np.exp2(shift).astype(dtype)
          b1 /= c
          b2 *= c[:, None]
        w1.append(b1)
        w2.append(b2)
        unit_task.append(np.full(len(units), i))

    def cat(blocks, axis, shape):
      return np.concatenate(blocks, axis=axis) if blocks else np.zeros(
          shape, dtype)

    layers.append(PackedLayer(
        wq=cat(wq, 1, (width, 0)), wk=cat(wk, 1, (width, 0)),
        wv=cat(wv, 1, (width, 0)), wo=cat(wo, 0, (0, width)),
        qk_sizes=tuple(qk_sizes), v_sizes=tuple(v_sizes),
        divisors=tuple(divisors), w1=cat(w1, 1, (width, 0)),
        w2=cat(w2, 0, (0, width)),
        unit_task=(np.concatenate(unit_task) if unit_task
                   else np.zeros(0, dtype=int))))
  if not np.isfinite(max_gate) or max_gate >= np.finfo(dtype).max / 4:
    raise ValueError(f"Gate {max_gate:.3e} does not fit in {dtype}.")

  packed = PackedModel(
      token_embed=te, pos_embed=pe, is_pad=is_pad, layers=layers, tasks=tasks,
      causal=progs[0].model.model_config.causal, slot_labels=slot_labels)
  d_programs = [int(p.dims.sum()) for p in progs]
  cols = sum(l.wq.shape[1] + l.wk.shape[1] + l.wv.shape[1] + l.wo.shape[0]
             for l in layers)
  n_units = sum(l.w1.shape[1] for l in layers)
  emb_rows = te.shape[0] + pe.shape[0]
  d_block = sum(d_programs)
  packed.stats = PackStats(
      programs=k, d_model=width, d_programs=d_programs, gate_slots=k,
      position_slots=len(pos_labels), slots_lower_bound=lower,
      slots_dsatur=dsatur_slots, slots_final=shared, solver=solver,
      ilp_status=status, heads=sum(len(l.v_sizes) for l in layers),
      mlp_units=n_units, params=packed.num_params,
      params_block_diagonal=d_block * (cols + 2 * n_units + emb_rows),
      max_gate=max_gate,
      max_weight=max(float(np.abs(w).max(initial=0.0)) for l in layers
                     for w in (l.wq, l.wk, l.wv, l.wo, l.w1, l.w2)))
  return packed

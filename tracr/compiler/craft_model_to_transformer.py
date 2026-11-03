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
"""Convert craft model into transformer with the correct input/output spaces."""

import networkx as nx
import numpy as np
from tracr.compiler import assemble
from tracr.compiler import nodes
from tracr.craft import bases
from tracr.craft import transformers
from tracr.craft.chamber import binary_mlp
from tracr.rasp import rasp
from tracr.transformer import encoder


def craft_model_to_transformer(
    craft_model: transformers.SeriesWithResiduals,
    graph: nx.DiGraph,
    sink: nodes.Node,
    max_seq_len: int,
    compiler_bos: str,
    compiler_pad: str,
    causal: bool = False,
) -> assemble.AssembledTransformerModel:
  """Turn a craft model into a transformer model."""

  if rasp.tokens.label not in graph.nodes:
    raise ValueError(
        f'Failed to find a node with label {rasp.tokens.label}. '
        'This is probably because your RASP program does not include '
        'rasp.tokens. A program must include rasp.tokens to be '
        'compiled.'
    )

  # Add the compiler BOS token.
  tokens_value_set = (
      graph.nodes[rasp.tokens.label][nodes.VALUE_SET].union(
          {compiler_bos, compiler_pad}))
  # Token ids are the positions in this basis, whatever the encoding.
  token_ids_space = bases.VectorSpaceWithBasis.from_values(rasp.tokens.label,
                                                           tokens_value_set)
  tokens_space, token_matrix = token_ids_space, None
  tokens_node = graph.nodes[rasp.tokens.label]
  if rasp.is_binary(tokens_node[nodes.EXPR]):
    tokens_space, token_matrix = _binary_token_embedding(
        tokens_node[nodes.BINARY_CODE], token_ids_space, compiler_bos)

  indices_space = bases.VectorSpaceWithBasis.from_values(
      rasp.indices.label, range(max_seq_len))
  index_matrix = None
  indices_node = graph.nodes.get(rasp.indices.label)
  if indices_node is not None and rasp.is_binary(indices_node[nodes.EXPR]):
    code = indices_node[nodes.BINARY_CODE]
    indices_space = code.space
    index_matrix = code.matrix[[code.index[i] for i in range(max_seq_len)]]

  sink_expr = sink[nodes.EXPR]
  categorical_output = not rasp.is_numerical(sink_expr)
  output_space = bases.VectorSpaceWithBasis(sink[nodes.OUTPUT_BASIS])
  output_readout, output_basis = None, output_space.basis
  if rasp.is_binary(sink_expr):
    # Score of output value t is code(t) . x, maximal only at the exact code.
    code = sink[nodes.BINARY_CODE]
    output_readout = code.matrix.T
    output_basis = [bases.BasisDirection(sink_expr.label, v)
                    for v in code.values]

  input_encoder = encoder.CategoricalEncoder(
      basis=token_ids_space.basis,
      enforce_bos=compiler_bos is not None,
      bos_token=compiler_bos,
      pad_token=compiler_pad,
      max_seq_len=max_seq_len + 1 if compiler_bos is not None else max_seq_len,
  )

  assembled_model = assemble.assemble_craft_model(
      craft_model=craft_model,
      tokens_space=tokens_space,
      indices_space=indices_space,
      output_space=output_space,
      categorical_output=categorical_output,
      causal=causal,
      # Mask PAD positions; the id must match the one the encoder emits.
      pad_token=input_encoder.pad_encoding,
      token_matrix=token_matrix,
      index_matrix=index_matrix,
      output_readout=output_readout,
  )

  assembled_model.input_encoder = input_encoder

  if categorical_output:
    assembled_model.output_encoder = encoder.CategoricalEncoder(
        basis=output_basis,
        enforce_bos=False,
        bos_token=None,
        pad_token=None)
  else:
    assembled_model.output_encoder = encoder.NumericalEncoder()

  return assembled_model


def _binary_token_embedding(
    code: binary_mlp.BinaryCode,
    token_ids_space: bases.VectorSpaceWithBasis,
    compiler_bos: str,
):
  """Tokens space and [num ids, dims] embedding for bit-coded tokens.

  A vocab token embeds as its code. BOS keeps its own one-hot direction (the
  attention heads key on it) and zero bits; PAD embeds as zero (it is masked).
  """
  bos_dir = bases.BasisDirection(rasp.tokens.label, compiler_bos)
  tokens_space = bases.join_vector_spaces(
      code.space, bases.VectorSpaceWithBasis([bos_dir]))
  index = tokens_space.index_by_direction
  bit_cols = [index[d] for d in code.space.basis]
  matrix = np.zeros((token_ids_space.num_dims, tokens_space.num_dims))
  for row, direction in enumerate(token_ids_space.basis):
    if direction == bos_dir:
      matrix[row, index[bos_dir]] = 1.0
    elif direction.value in code.index:
      matrix[row, bit_cols] = code.code(direction.value)
  return tokens_space, matrix

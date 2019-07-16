# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from models.Dense2D import Dense_without_bias


class MultiHeadAttentionLayer(tl.models.Model):
  """Multi-headed attention layer."""

  def __init__(self, num_heads, hidden_size, keep_pro):
    """Initialize Attention.

    Args:
      hidden_size: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      attention_dropout: float, dropout rate inside attention for training.
    """
    if hidden_size % num_heads:
      raise ValueError(
          "Hidden size ({}) must be divisible by the number of heads ({})."
          .format(hidden_size, num_heads))

    super(MultiHeadAttentionLayer, self).__init__()
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.attention_dropout = 1-keep_pro
    # Layers for linearly projecting the queries, keys, and values.
    # self.q_dense_layer = Dense_without_bias(
    #   self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="q")
    # self.k_dense_layer = Dense_without_bias(
    #   self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="k")
    # self.v_dense_layer = Dense_without_bias(
    #   self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="v")
    self.output_dense_layer = Dense_without_bias(
      self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="output_transform")
    
    self.w_layer = Dense_without_bias(self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="q")


  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }

  def split_heads(self, x):
    """Split x into different heads, and transpose the resulting value.

    The tensor is transposed to insure the inner dimensions hold the correct
    values during the matrix multiplication.

    Args:
      x: A tensor with shape [batch_size, length, hidden_size]

    Returns:
      A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
    """
    with tf.name_scope("split_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[1]

      # Calculate depth of last dimension after it has been split.
      depth = (self.hidden_size // self.num_heads)

      # Split the last dimension
      x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

      # Transpose the result
      return tf.transpose(x, [0, 2, 1, 3])

  def combine_heads(self, x):
    """Combine tensor that has been split.

    Args:
      x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

    Returns:
      A tensor with shape [batch_size, length, hidden_size]
    """
    with tf.name_scope("combine_heads"):
      batch_size = tf.shape(x)[0]
      length = tf.shape(x)[2]
      x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
      return tf.reshape(x, [batch_size, length, self.hidden_size])

  def forward(self, x, y, mask, cache=None):
    """Apply attention mechanism to x and y.

    Args:
      x: a tensor with shape [batch_size, length_x, hidden_size]
      y: a tensor with shape [batch_size, length_y, hidden_size]
      bias: attention bias that will be added to the result of the dot product.
      training: boolean, whether in training mode or not.
      cache: (Used during prediction) dictionary with tensors containing results
        of previous attentions. The dictionary must have the items:
            {"k": tensor with shape [batch_size, i, key_channels],
             "v": tensor with shape [batch_size, i, value_channels]}
        where i is the current decoded length.

    Returns:
      Attention layer output with shape [batch_size, length_x, hidden_size]
    """
    # Linearly project the query (q), key (k) and value (v) using different
    # learned projections. This is in preparation of splitting them into
    # multiple heads. Multi-head attention uses multiple queries, keys, and
    # values rather than regular attention (which uses a single q, k, v).
    bias = mask
    Batch_size = x.shape[0]
    x = tf.reshape(x, [-1, x.shape[-1]])
    q = self.w_layer(x)
    q = tf.reshape(q, [Batch_size, -1, q.shape[-1]])
    k = v = y
    # y = tf.reshape(y, [-1, y.shape[-1]])
    # q = self.q_dense_layer(x)
    # k = self.k_dense_layer(y)
    # v = self.v_dense_layer(y)
    # q = tf.reshape(q, [Batch_size, -1, q.shape[-1]])
    # k = tf.reshape(k, [Batch_size, -1, k.shape[-1]])
    # v = tf.reshape(v, [Batch_size, -1, v.shape[-1]])

    if cache is not None:
      
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    # Split q, k, v into heads.
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v) #(Batch, num_head, length_v, dk)
    
    # Scale q to prevent the dot product between q and k from growing too large.
    depth = (self.hidden_size // self.num_heads)
    q *= depth ** -0.5
    
    # Calculate dot product attention
    logits = tf.matmul(q, k, transpose_b=True) #(Batch, num_head, length_q, length_k)
    logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights") #(Batch, num_head, length_q, length_k)
    if self.is_train:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    
    attention_output = tf.matmul(weights, v)

    # Recombine heads --> [batch_size, length_q, hidden_size]
    attention_output = self.combine_heads(attention_output)

    # Run the combined outputs through another linear projection layer.
    attention_output = tf.reshape(attention_output, [-1, attention_output.shape[-1]])
    attention_output = self.output_dense_layer(attention_output)
    attention_output = tf.reshape(attention_output, [Batch_size, -1, attention_output.shape[-1]])
    return attention_output


class SelfAttentionLayer(MultiHeadAttentionLayer):
  """Multiheaded self-attention layer."""

  def forward(self, inputs, mask, cache=None):
    x = inputs
    return super(SelfAttentionLayer, self).forward(x, x, mask, cache)
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
import numpy as np


class MultiHeadAttentionLayer(tl.models.Model):
  """Multi-headed attention layer."""

  def __init__(self, num_heads, hidden_size, keep_prob):
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
    self.attention_dropout = 1-keep_prob
    self.seq_length = 50

    self.one_by_one_layer = tl.layers.Conv1d(
              n_filter=1,
              filter_size=2,
              stride=2,
              padding='VALID',
              in_channels=hidden_size
          )
    # for i in range(self.seq_length):
    #   self.one_by_one_layer.append(tl.layers.Conv1d(
    #           n_filter=1,
    #           filter_size=2,
    #           stride=2,
    #           padding='VALID',
    #           in_channels=hidden_size
    #       )
    #   )
    self.output_dense_layer = Dense_without_bias(
      self.hidden_size, in_channels=self.hidden_size, W_init=tf.keras.initializers.get('glorot_uniform'), name="output_transform")
    


  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "num_heads": self.num_heads,
        "attention_dropout": self.attention_dropout,
    }


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
    k = v = y
    q = x
    
    for i in range(y.shape[1]):
        concat = np.zeros((Batch_size,x.shape[1]+y.shape[1],x.shape[-1]))
        concat[:,::2,:] = x
        concat[:,1::2,:] = y
        concat = tf.convert_to_tensor(concat, dtype=tf.float32)
        y = tf.roll(y, shift=[-1], axis=[1])
        concat = self.one_by_one_layer(concat)
        if (i == 0):
            output = concat
        else:
            output = tf.concat([output, concat], axis=2)


    for i in range(x.shape[1]):
        out = tf.reshape(output[:,i,:], [Batch_size, 1, -1])
        seq = tf.roll(out, shift=[i], axis=[2])
        if (i == 0):
          output_ = seq
        else:
          output_ = tf.concat([output_, seq], axis=1)
    
    attention = output_ # [Batch_size, length, length]
    bias = tf.squeeze(bias, 1)
    attention += bias
    weights = tf.nn.softmax(attention, name="attention_weights") #(Batch, num_head, length_q, length_k)
    if self.is_train:
      weights = tf.nn.dropout(weights, rate=self.attention_dropout)
    
    attention_output = tf.matmul(weights, v)
    

    if cache is not None:
      
      # Combine cached keys and values with new keys and values.
      k = tf.concat([cache["k"], k], axis=1)
      v = tf.concat([cache["v"], v], axis=1)

      # Update cache
      cache["k"] = k
      cache["v"] = v

    



    # Run the combined outputs through another linear projection layer.
    attention_output = tf.reshape(attention_output, [-1, attention_output.shape[-1]])
    attention_output = self.output_dense_layer(attention_output)
    attention_output = tf.reshape(attention_output, [Batch_size, -1, attention_output.shape[-1]])

    print(attention_output.shape)
    return attention_output


class SelfAttentionLayer(MultiHeadAttentionLayer):
  """Multiheaded self-attention layer."""

  def forward(self, inputs, mask, cache=None):
    x = inputs
    return super(SelfAttentionLayer, self).forward(x, x, mask, cache)

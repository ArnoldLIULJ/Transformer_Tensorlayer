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
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl


class FeedForwardNetwork(tl.models.Model):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, keep_prob):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(FeedForwardNetwork, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = 1-keep_prob
    self.one_by_one_layer = tl.layers.Conv1d(
            n_filter=hidden_size,
            filter_size=1,
            stride=1,
            padding='VALID',
            in_channels=hidden_size
        )

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "filter_size": self.filter_size,
        "relu_dropout": self.relu_dropout,
    }

  def forward(self, inputs):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    # Retrieve dynamically known shapes
    x = inputs
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    # x = tf.reshape(x, [-1, x.shape[-1]])
    output = self.one_by_one_layer(x)
    output = tf.nn.relu(output)
    # output = tf.reshape(output, [batch_size, -1, output.shape[-1]])
    if self.is_train:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    # output = tf.reshape(output, [-1, output.shape[-1]])
    # output = self.output_dense_layer(output)
    # output = tf.reshape(output, [batch_size, -1, output.shape[-1]])

    return output



class TuckerDecomposition_FeedForwardNetwork(tl.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, R1, R2):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(TuckerDecomposition_FeedForwardNetwork, self).__init__()
    self.I2 = hidden_size
    self.I1 = filter_size
    self.R1 = R1
    self.R2 = R2
    self.relu_dropout = relu_dropout
    
    self.build(tuple())
    self._built = True

  def build(self, inputs_shape):
    with tf.name_scope("tucker_decomposition"):

        self.U1 = self._get_weights('U1', shape=(self.I1, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))
        self.U2 = self._get_weights('U2', shape=(self.I2, self.R2),
            init=tf.random_normal_initializer(mean=0., stddev=self.R2**-0.5))
        self.G = self._get_weights('G', shape=(self.R2, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))

        # self.U1 = self._get_weights('U1', shape=(self.I1, self.R1),
        #     init=tf.random_normal_initializer(mean=0., stddev=self.hidden_size**-0.5))
        # self.U2 = self._get_weights('U1', shape=(self.I2, self.R2),
        #     init=tf.random_normal_initializer(mean=0., stddev=self.hidden_size**-0.5))
        # self.G = self._get_weights('G', shape=(self.R2, self.R1),
        #     init=tf.random_normal_initializer(mean=0., stddev=self.hidden_size**-0.5))


  def forward(self, inputs):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    # Retrieve dynamically known shapes
    x = inputs
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    x = tf.reshape(x, [-1, x.shape[-1]])
    w = tf.matmul(self.U2, self.G)
    w = tf.matmul(w, tf.transpose(self.U1))
    output = tf.matmul(x, w)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [batch_size, -1, output.shape[-1]])
    if self.is_train:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = tf.reshape(output, [-1, output.shape[-1]])
    w = tf.matmul(self.U1, tf.transpose(self.G))
    w = tf.matmul(w, tf.transpose(self.U2))
    output = tf.matmul(output, w)
    # output = self.output_dense_layer(output)
    output = tf.reshape(output, [batch_size, -1, output.shape[-1]])

    return output






class TuckerDecomposition_FeedForwardNetwork_2(tl.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout, R1, R2):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(TuckerDecomposition_FeedForwardNetwork_2, self).__init__()
    self.I2 = hidden_size
    self.I1 = filter_size
    self.R1 = R1
    self.R2 = R2
    self.relu_dropout = relu_dropout
    
    self.build(tuple())
    self._built = True

  def build(self, inputs_shape):
    with tf.name_scope("tucker_decomposition"):

        self.U1 = self._get_weights('U1', shape=(self.I1, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))
        self.U2 = self._get_weights('U2', shape=(self.I2, self.R2),
            init=tf.random_normal_initializer(mean=0., stddev=self.R2**-0.5))
        self.G = self._get_weights('G', shape=(self.R2, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))

        self.U1_ = self._get_weights('U1_', shape=(self.I1, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))
        self.U2_ = self._get_weights('U2_', shape=(self.I2, self.R2),
            init=tf.random_normal_initializer(mean=0., stddev=self.R2**-0.5))
        self.G_ = self._get_weights('G_', shape=(self.R2, self.R1),
            init=tf.random_normal_initializer(mean=0., stddev=self.R1**-0.5))


  def forward(self, inputs):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    # Retrieve dynamically known shapes
    x = inputs
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]
    x = tf.reshape(x, [-1, x.shape[-1]])
    w = tf.matmul(self.U2, self.G)
    w = tf.matmul(w, tf.transpose(self.U1))
    output = tf.matmul(x, w)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [batch_size, -1, output.shape[-1]])
    if self.is_train:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = tf.reshape(output, [-1, output.shape[-1]])
    w = tf.matmul(self.U1_, tf.transpose(self.G_))
    w = tf.matmul(w, tf.transpose(self.U2_))
    output = tf.matmul(output, w)
    # output = self.output_dense_layer(output)
    output = tf.reshape(output, [batch_size, -1, output.shape[-1]])

    return output

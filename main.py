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
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from models.transformer_v2 import Transformer
from models import model_params





class gan_transformer(tl.models.Model):
  def __init__(self, params, name=None):
    super(gan_transformer, self).__init__(name=name)
    self.genearator = Transformer(params)
    self.discriminator = tl.layers.Dense(n_units=1, in_channels=params.vocab_size, act=tf.nn.sigmoid)

  def forward(self, inputs):
    input = inputs[0]
    target = inputs[1]
    length = target.shape[1]
    fake = self.genearator(input, targets = target)
    fake = tf.reshape(fake, [-1, fake.shape[-1]])
    output = self.discriminator(fake)
    output = tf.reshape(output, [-1, length, 1])
    return fake, output


params = model_params.TINY_PARAMS

input = tl.layers.Input([80, 20], name='input', dtype=tf.int32)
target = tl.layers.Input([80, 20], name='input', dtype=tf.int32)

model = gan_transformer(params)
model.train()
print(model([input, target]))
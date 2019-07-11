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
from models.Discriminator_v2 import Discriminator






class gan_transformer(tl.models.Model):
  def __init__(self, params, name=None):
    self.generator = Transformer(params)
    self.discriminator = Discriminator(params)
    super(gan_transformer, self).__init__(name=name)

  def forward(self, input, targets):
    input = input
    target = targets
    fake = self.generator(input, targets = target)
    output = self.discriminator(fake)
    return fake, output





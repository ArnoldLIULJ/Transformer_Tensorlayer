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
from weightLightModels.LightWeightConvolution import LightConv
from models import embedding_layer_v2 as embedding_layer





class Discriminator(tl.models.Model):
  def __init__(self, params, name=None):
    self.input_layer = tl.layers.Dense(n_units=params.hidden_size, in_channels=params.vocab_size)
    self.conv_layer = LightConv(params)
    super(Discriminator, self).__init__(name=name)

  def forward(self, input):

    batch_size = input.shape[0]
    input = tf.reshape(input, [-1, input.shape[-1]])
    input = self.input_layer(input)
    input = tf.reshape(input, [batch_size, -1, input.shape[-1]])
    output = self.conv_layer(input)
    # print(output.shape)
    

    return output
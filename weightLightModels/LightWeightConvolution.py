import tensorlayer as tl
import tensorflow as tf
from weightLightModels.GatedLinearUnit import GLU
from weightLightModels.LightConv import LConv


class LightConv(tl.models.Model):

    def __init__(self, params, name=None):

        super(LightConv, self).__init__(name=name)
        print("It is Light Version")
        self.params = params
        self.in_layer = tl.layers.Dense(n_units=params.n_units, in_channels=params.hidden_size)
        self.glu_layer = GLU(params)
        self.light_conv_layer = LConv(params)
        self.out_layer = tl.layers.Dense(n_units=params.hidden_size, in_channels=params.filter_number//2)

    def forward(self, inputs):
        Batch_size = inputs.shape[0]
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        inputs = self.in_layer(inputs)
        inputs = tf.reshape(inputs, [Batch_size, -1, inputs.shape[-1]])
        inputs = self.glu_layer(inputs)
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = self.light_conv_layer(inputs)
        inputs = tf.squeeze(inputs, axis=1)
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        inputs = self.out_layer(inputs)
        inputs = tf.reshape(inputs, [Batch_size, -1, inputs.shape[-1]])

        return inputs





    


    
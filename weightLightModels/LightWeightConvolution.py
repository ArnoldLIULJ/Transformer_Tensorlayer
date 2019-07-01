import tensorlayer as tl
import tensorflow as tf
from weightLightModels.GatedLinearUnit import GLU
from weightLightModels.LightConv import LConv


class LightConv(tl.models.Model):

    def __init__(self, params, padding='VALID', name=None):

        super(LightConv, self).__init__(name=name)
        self.params = params
        self.in_layer = tl.layers.Dense(n_units=params.n_units, in_channels=params.hidden_size)
        self.glu_layer = GLU(params)
        self.light_conv_layer = LConv(params, padding)
        self.out_layer = tl.layers.Dense(n_units=params.hidden_size, in_channels=params.filter_number//2)

    def forward(self, inputs, cache=None):
        Batch_size = inputs.shape[0]
        original_length = inputs.shape[1]

        if cache is not None:
            inputs = tf.concat([cache["ids"], inputs], axis=1)
            cache["ids"] = inputs
        
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        inputs = self.in_layer(inputs)

        inputs = tf.reshape(inputs, [Batch_size, -1, inputs.shape[-1]])
        inputs = tf.pad(inputs, [[0,0],[self.params.filter_size-1, 0],[0,0]])
        inputs = self.glu_layer(inputs)
        inputs = tf.pad(inputs, [[0,0],[self.params.light_filter_size[1]-1, 0],[0,0]])
        # reshape inputs to [B, 1, S, H]
        inputs = tf.expand_dims(inputs, axis=1)
        inputs = self.light_conv_layer(inputs)

        inputs = tf.squeeze(inputs, axis=1)
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
        inputs = self.out_layer(inputs)
        inputs = tf.reshape(inputs, [Batch_size, -1, inputs.shape[-1]])
        inputs = inputs[:,-original_length:,:]
        return inputs





    


    
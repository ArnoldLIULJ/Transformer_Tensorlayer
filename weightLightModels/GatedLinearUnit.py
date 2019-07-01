import tensorflow as tf
import tensorlayer as tl
from weightLightModels.AddBiasLayer import addBias


class GLU(tl.models.Model):
    def __init__(self, params, name=None):
        super(GLU, self).__init__(name=name)
        self.params = params
        self.conv_layer = tl.layers.Conv1d(
            n_filter=params.filter_number,
            filter_size=params.filter_size,
            stride=1,
            padding='VALID',
            in_channels=params.n_units
        )
        self.add_bias_layer = addBias(
            in_channels=params.filter_number
        )
        

    def forward(self, inputs):
        inputs = self.conv_layer(inputs)
        inputs = self.add_bias_layer(inputs)
        A = inputs[:,:,:inputs.shape[-1]//2]
        B = inputs[:,:,inputs.shape[-1]//2:]
        inputs = A * tf.nn.sigmoid(B)
        return inputs




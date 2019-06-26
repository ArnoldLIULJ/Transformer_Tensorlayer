import tensorflow as tf
import tensorlayer as tl

class addBias(tl.layers.Layer):
    def __init__(self, in_channels):
        super(addBias, self).__init__()
        self.in_channels = in_channels
        self.build(tuple())
        self._built = True

    def build(self, inputs_shape):
        self.bias = self._get_weights('bias', shape=(1,1,self.in_channels))
    
    def forward(self, inputs):
        inputs = inputs + self.bias
        return inputs


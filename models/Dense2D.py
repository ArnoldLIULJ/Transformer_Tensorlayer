import tensorflow as tf
import tensorlayer as tl

class Dense_without_bias(tl.layers.Layer):


  def __init__(self, hidden_size, in_channels, W_init, name=None):

    super(Dense_without_bias, self).__init__(name=name)
    self.hidden_size = hidden_size
    self.in_channel = in_channels
    self.w_init = W_init
    self.build(tuple())
    self._built = True

  def build(self, inputs_shape):
    with tf.name_scope("Dense_without_bias"):
        self.W = self._get_weights('weights', shape=(self.in_channel, self.hidden_size),
            init=self.w_init)

  def forward(self, inputs):
      return tf.matmul(inputs, self.W)

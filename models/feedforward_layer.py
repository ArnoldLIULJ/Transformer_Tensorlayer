import tensorlayer as tl
import tensorflow as tf


class FeedForwardLayer(tl.layers.Layer):
    """
    Feed forward layer

    Parameters
    ----------
    hidden_size:
        hidden size of both input and output
    ff_size:
        hidden size used in the middle layer of this feed forward layer
    keep_prob:
        keep probability of dropout layer

    """

    def __init__(self, hidden_size, ff_size, keep_prob):
        super(FeedForwardLayer, self).__init__()
        self.hidden_size = hidden_size
        self.ff_size = ff_size
        self.keep_prob = keep_prob

        self._nodes_fixed = True
        if not self._built:
            self.build(tuple())
            self._built = True

    def build(self, inputs_shape):
        # self.dense1 = tl.layers.Dense(self.ff_size)
        # self.dense2 = tl.layers.Dense(self.hidden_size)
        # self.dropout = tl.layers.Dropout(self.keep_prob)
        self.W1 = self._get_weights('W1', (self.hidden_size, self.ff_size))
        self.W2 = self._get_weights('W2', (self.ff_size, self.hidden_size))

    def forward(self, inputs):
        # print(inputs.shape)
        # return self.dense2(self.dropout(tf.nn.relu(self.dense1(inputs))))
        out = tf.tensordot(inputs, self.W1, axes=[[2], [0]])
        out = tf.nn.relu(out)
        if self.is_train:
            out = tf.nn.dropout(out, rate=1-self.keep_prob)
        out = tf.tensordot(out, self.W2, axes=[[2], [0]])
        return out

    def __repr__(self):
        return "feedforward layer"

'''

class FeedForwardNetwork_tf(tl.layers.Layer):
  """Fully connected feedforward network."""

  def __init__(self, hidden_size, filter_size, relu_dropout):
    """Initialize FeedForwardNetwork.

    Args:
      hidden_size: int, output dim of hidden layer.
      filter_size: int, filter size for the inner (first) dense layer.
      relu_dropout: float, dropout rate for training.
    """
    super(FeedForwardNetwork_tf, self).__init__()
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.relu_dropout = relu_dropout

  def build(self, input_shape):
    self.filter_dense_layer = tf.keras.layers.Dense(
        self.filter_size,
        use_bias=True,
        activation=tf.nn.relu,
        name="filter_layer")
    self.output_dense_layer = tf.keras.layers.Dense(
        self.hidden_size, use_bias=True, name="output_layer")
    super(FeedForwardNetwork_tf, self).build(input_shape)

  def get_config(self):
    return {
        "hidden_size": self.hidden_size,
        "filter_size": self.filter_size,
        "relu_dropout": self.relu_dropout,
    }

  def call(self, x, training):
    """Return outputs of the feedforward network.

    Args:
      x: tensor with shape [batch_size, length, hidden_size]
      training: boolean, whether in training mode or not.

    Returns:
      Output of the feedforward network.
      tensor with shape [batch_size, length, hidden_size]
    """
    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    output = self.filter_dense_layer(x)
    if training:
      output = tf.nn.dropout(output, rate=self.relu_dropout)
    output = self.output_dense_layer(output)

    return output

    '''
import tensorlayer as tl
import tensorflow as tf


class EmbeddingLayer(tl.layers.Layer):
    """
    Embedding layer

    Parameters:
        vocab_size: vocabulary size
        hidden_size: embedding size, the output size of each word after embedding
    """

    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.build(tuple())
        self._built = True

    def __repr__(self):
        return "embedding"

    def build(self, inputs_shape):
        self.W = self._get_weights('weights', shape=(self.vocab_size, self.hidden_size),init=tf.random_normal_initializer(
              mean=0., stddev=self.hidden_size**-0.5))

    def forward(self, inputs):
        # inputs is of size (batch_size, length)
        # create mask for inputs, 0 is <pad> in dictionary
        if (len(inputs.shape) == 3):
            return self._linear(inputs)
        mask = tf.cast(tf.not_equal(inputs, 0), dtype=tf.float32)

        embeddings = tf.gather(self.W, inputs)
        embeddings *= tf.expand_dims(mask, 2)
        embeddings *= self.hidden_size ** 0.5

        return embeddings
    
    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.

        Args:
        inputs: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
        float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]

        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.W, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])

import tensorflow as tf
import tensorlayer as tl
class example(tl.layers.Layer):


  def __init__(self, name=None):

    super(example, self).__init__(name=name)
    self.input_layer = []
    for i in range(4):
      self.input_layer.append(tl.layers.Dense(in_channels=50, n_units=50))

    self.input_layers = tl.layers.LayerList(self.input_layer)

    self.build(None)
    self._built = True

  def build(self, inputs_shape=None):
    self.W = self._get_weights('weights', shape=(50, 10))
    
  def forward(self, inputs):
      for layer in self.input_layers:
        inputs = layer(inputs)
      output = tf.matmul(inputs, self.W)
      return output

class model(tl.models.Model):
    def __init__(self, name=None):
      super(model, self).__init__(name=name)
      self.layer = example()
    def forward(self, inputs):
      return self.layer(inputs)


input = tf.random.normal(shape=(100,50))
model_ = model()
model_.train()
print(model_(input))
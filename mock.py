

import tensorflow as tf
import tensorlayer as tl
class example(tl.layers.Layer):


  def __init__(self, name=None):

    super(example, self).__init__(name=name)
    self.input_layer = tl.layers.Dense(in_channels=50, n_units=20)
    self.build(None)
    self._built = True

  def build(self, inputs_shape=None):
    self.W = self._get_weights('weights', shape=(20, 10))
    
  def forward(self, inputs):
      inputs = self.input_layer(inputs)
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
with tf.GradientTape() as tape:
    out = model_(input)
    target = input[:,:10]
    loss = out-target 
    grad = tape.gradient(loss, model_.all_weights)





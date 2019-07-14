

import tensorflow as tf
import tensorlayer as tl
import numpy as np

x = tl.layers.Input([1,6,2], dtype=tf.float64, name='input')
y = tl.layers.Input([1,6,2], dtype=tf.float64, name='input_2')
seq_length=6

one_by_one_layer = []
for i in range(6):
  one_by_one_layer.append(tl.layers.Conv1d(
          n_filter=1,
          filter_size=2,
          stride=2,
          padding='VALID',
          in_channels=2
      )
  )
concat = np.zeros((1,12,2))
output = []
for i in range(y.shape[1]):
    concat[:,::2,:] = x
    concat[:,1::2,:] = y
    y = tf.roll(y, shift=[1], axis=[1])
    concat = tf.convert_to_tensor(concat, dtype=tf.float64)
    output.append(one_by_one_layer[i](concat))

output = tf.convert_to_tensor(output)
# print(output)
output = tf.reshape(output, [1, 6, 6, 2])
# print(output)
output = tf.tensordot(output, y, axes=[[2],[2]])
print(output)

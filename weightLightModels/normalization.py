from weightLightModels.wrapper import Wrapper
from tensorlayer.layers import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.

    Input:
        A tensor
        
    Output:
        The output of layer by weight normalization

    Parameters
    ------------
    layer :
        inherited from tensorlayer.layers

    Examples
    ---------
    With TensorLayer

    
    >>> net = tl.layers.Input([200, 32], name='input')
    >>> layer = WeightNorm(tl.layers.Dense(n_units=5, in_channels=32))
    >>> print(layer(net))
    >>> output shape : (200, 5)

    >>> net = tl.layers.Input([8, 200, 200, 32], name='input')
    >>> layer = WeightNorm(DepthwiseConv2d(
    ...            filter_size=(5,5),
    ...            in_channels=32), padding=padding)
    >>> print(layer(net))
    >>> output shape : (8, 200, 200, 32)


    References
    -----------
    - tensorflow_addons : https://github.com/tensorflow/addons/blob/master/tensorflow_addons/layers/wrappers.py
    """
    def __init__(self, layer, **kwargs):
        super(WeightNorm, self).__init__(layer, **kwargs)
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))


    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.W = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if not self.layer._built:
            self.layer.build(input_shape)
        if not hasattr(self.layer, 'W'):
            raise ValueError(
                '`WeightNorm` must wrap a layer that'
                ' contains a `W` for weights'
            )

        # The kernel's filter or unit dimension is -1
        self.layer_depth = int(self.layer.W.shape[-1])
        self.norm_axes = list(range(self.layer.W.shape.ndims - 1))

        self.layer.v = self.layer.W
        self.layer.g = self.layer._get_weights(
            var_name="g",
            shape=(self.layer_depth,),
            init=initializers.get('ones'),
            trainable=True)
        
        with ops.control_dependencies([self.layer.g.assign(
                self._init_norm(self.layer.v))]):
            self._compute_weights()

        self.layer._built = True

        super(WeightNorm, self).build()
        self._built = True

    def forward(self, inputs):
        """Call `Layer`"""
        self._compute_weights()  # Recompute weights for each forward pass
        output = self.layer(inputs)
        
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

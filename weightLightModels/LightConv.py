import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers.convolution.depthwise_conv import DepthwiseConv2d
from weightLightModels.normalization import WeightNorm
class LConv(tl.models.Model):
    def __init__(self, params, padding='VALID', name=None):
        super(LConv, self).__init__(name=name)
        self.params = params
        self.H = params.H
        self.in_channels = params.filter_number//2
        # self.softmax_layer = tl.layers.Lambda(lambda x: tf.nn.softmax(x), name='lambda')
        self.conv_Layer = WeightNorm(DepthwiseConv2d(
                filter_size=params.light_filter_size,
                in_channels=self.H, padding=padding), in_channels=self.H)
    
    def forward(self, inputs):

        for i in range(self.in_channels//self.H):
            inputs_sliced = inputs[:,:,:,i*self.H:(i+1)*self.H]
            outputs_sliced = self.conv_Layer(inputs_sliced)
            if i == 0:
                output = outputs_sliced
            else:
                output = tf.concat([output, outputs_sliced], axis=3)
        
        return output 







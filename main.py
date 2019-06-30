import tensorflow as tf
import tensorlayer as tl
from weightLightModels.model_params import TINY_PARAMS
from weightLightModels.GatedLinearUnit import GLU
from weightLightModels.LightConv import LConv
from weightLightModels.LightWeightConvolution import LightConv
input = tl.layers.Input([8, 1, 64], name='input')
net = LightConv(TINY_PARAMS, name="LConv")
net.train()
print(net(input))
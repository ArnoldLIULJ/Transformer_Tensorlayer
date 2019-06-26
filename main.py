import tensorflow as tf
import tensorlayer as tl
from weightLightModels.models_params import TINY_PARAMS
from weightLightModels.GatedLinearUnit import GLU
from weightLightModels.LightConv import LConv
from weightLightModels.LightWeightConvolution import LightConv
input = tl.layers.Input([8, 100, 16], name='input')
net = LightConv(TINY_PARAMS, name="LConv")
net.train()
print(net(input))
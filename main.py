import tensorflow as tf
import tensorlayer as tl
from weightLightModels.models_params import TINY_PARAMS
from weightLightModels.GatedLinearUnit import GLU

input = tl.layers.Input([8, 100, 16], name='input')
net = GLU(TINY_PARAMS)
net.train()
print(net(input))
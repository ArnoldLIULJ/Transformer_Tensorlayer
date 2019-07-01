#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from v2.transformer import Transformer
from v2.models_params import TINY_PARAMS as params
from tests.utils import CustomTestCase
from tensorlayer.cost import cross_entropy_seq
from utils import metrics
from models import optimizer
import time

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=5):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class Model_SEQ2SEQ_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 16

        cls.embedding_size = 32
        cls.trainX = np.random.randint(low=2, high=50, size=(50, 10))
        cls.trainY = np.random.randint(low=2, high=50, size=(50, 10))

        cls.trainX[:,-1] = 1
        cls.trainY[:,-1] = 1
        # Parameters
        cls.src_len = len(cls.trainX)
        cls.tgt_len = len(cls.trainY)

        assert cls.src_len == cls.tgt_len

        cls.num_epochs = 1000
        cls.n_step = cls.src_len // cls.batch_size

    @classmethod
    def tearDownClass(cls):
        pass

    def test_basic_simpleSeq2Seq(self):
        model_ = Transformer(params)
        self.vocab_size = params["vocab_size"]

        # optimizer_ = optimizer.LazyAdam(
        #     params["learning_rate"],
        #     params["optimizer_adam_beta1"],
        #     params["optimizer_adam_beta2"],
        #     epsilon=params["optimizer_adam_epsilon"])

        learning_rate = CustomSchedule(params["hidden_size"])
        optimizer_ = tf.optimizers.Adam(learning_rate=0.01)
        # optimizer_ = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
        #                              epsilon=1e-9)
        # optimizer_ = optimizer.LazyAdam(learning_rate, beta_1=0.9, beta_2=0.98, 
        #                              epsilon=1e-9)
        


        for epoch in range(self.num_epochs):
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):


                with tf.GradientTape() as tape:
                    
                    targets = Y
                    output = model_(inputs = [X, Y], training=True)
                    # print(len(model_.trainable_weights))
                    # print(model_.trainable_weights)
                    # exit()
                    # print(logits.shape, Y.shape)
                    logits = metrics.MetricLayer(self.vocab_size)([output, targets])
                    logits, loss = metrics.LossLayer(self.vocab_size, 0.1)([logits, targets])
                    # logits = tf.keras.layers.Lambda(lambda x: x, name="logits")(logits)
                    # print(time.time()-start)
                    # output = tf.reshape(output, [-1, output.shape[-1]])
                    # print(", ".join([t.name for t in model_.trainable_weights]))
                    # layer_normalization_print = [x for x in [t.name for t in model_.trainable_weights] if "feed_forward_network" in x ]
                    # print(", ".join(x for x in [t.name for t in model_.trainable_weights] if "feed_forward_network" in x ))
                    # print("number of layers : ", len(model_.trainable_weights))
                    # exit()
                    # loss = cross_entropy_seq(logits=output, target_seqs=Y)

                    grad = tape.gradient(loss, model_.trainable_weights)
                    # print(grad)
                    # exit()
                    optimizer_.apply_gradients(zip(grad, model_.trainable_weights))
                    # print(time.time()-start)
                total_loss += loss
                n_iter += 1
            
            test_sample = trainX[0:2, :]

            top_n = 1
            for i in range(top_n):
                prediction = model_(inputs = [test_sample], training=False)
                print("Prediction: >>>>>  ", prediction["outputs"], "\n Target: >>>>>  ", trainY[0:2, :], "\n\n")

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()

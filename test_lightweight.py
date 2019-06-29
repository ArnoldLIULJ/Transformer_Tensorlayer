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
# from models.transformer_v2 import Transformer
from weightLightModels.transformer import Transformer
from weightLightModels.models_params import TINY_PARAMS
# from models import model_params
from tests.utils import CustomTestCase
from utils import metrics
from tensorlayer.cost import cross_entropy_seq
from models import optimizer
import time



class Model_SEQ2SEQ_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 16

        cls.embedding_size = 32
        cls.dec_seq_length = 5
        cls.trainX = np.random.randint(low=2, high=50, size=(50, 10))
        cls.trainY = np.random.randint(low=2, high=50, size=(50, 11))

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
        model_ = Transformer(TINY_PARAMS)
        self.vocab_size = TINY_PARAMS.vocab_size
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        # optimizer = optimizer.LazyAdam(
        #     params["learning_rate"],
        #     params["optimizer_adam_beta1"],
        #     params["optimizer_adam_beta2"],
        #     epsilon=params["optimizer_adam_epsilon"])
        # print(model_.trainable_weights)
        # layer_normalization_print = [x for x in [t.name for t in model_.trainable_weights] if "feedforwardlayer" in x ]
        # print(", ".join([t.name for t in model_.trainable_weights]))
        # print(", ".join(layer_normalization_print))
        # print("number of layers :  ", len(model_.trainable_weights))
        # exit()
        # print(len(model_.trainable_weights))
        # print(model_.trainable_weights)
        # exit()
        for epoch in range(self.num_epochs):
            model_.train()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                with tf.GradientTape() as tape:

                    targets = Y
                    logits = model_(inputs = X, targets = Y)
                    logits = metrics.MetricLayer(self.vocab_size)([logits, targets])
                    logits, loss = metrics.LossLayer(self.vocab_size, 0.1)([logits, targets])
                    grad = tape.gradient(loss, model_.all_weights)
                    optimizer.apply_gradients(zip(grad, model_.all_weights))
                    
            
                total_loss += loss
                n_iter += 1
            model_.eval()
            test_sample = trainX[0:2, :]

            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()

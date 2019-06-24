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
from models.transformer import Transformer
from models import model_params
from tests.utils import CustomTestCase
from tensorlayer.cost import cross_entropy_seq




class Model_SEQ2SEQ_Test(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 16

        cls.vocab_size = 100
        cls.embedding_size = 32
        cls.dec_seq_length = 5
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
        model_ = Transformer(model_params.tiny_PARAMS)

        optimizer = tf.optimizers.Adam(learning_rate=0.001)
        # print(", ".join([t.name for t in model_.trainable_weights]))
        print(len(model_.trainable_weights), len(model_.all_weights))
        # exit()

        for epoch in range(self.num_epochs):
            model_.train()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):


                with tf.GradientTape() as tape:
                    
                    output = model_(inputs = X, targets = Y)
                    output = tf.reshape(output, [-1, output.shape[-1]])
                    

                    loss = cross_entropy_seq(logits=output, target_seqs=Y)

                    grad = tape.gradient(loss, model_.all_weights)
                    optimizer.apply_gradients(zip(grad, model_.all_weights))

                total_loss += loss
                n_iter += 1

            model_.eval()
            test_sample = trainX[0:2, :]

            top_n = 1
            for i in range(top_n):
                prediction = model_(inputs = test_sample)
                print("Prediction: >>>>>  ", prediction, "\n Target: >>>>>  ", trainY[0:2, :], "\n\n")

            # printing average loss after every epoch
            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()

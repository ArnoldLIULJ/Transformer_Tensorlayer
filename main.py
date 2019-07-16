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
from models.GAN import gan_transformer as Transformer
from models.model_params import TINY_PARAMS
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
        generator = model_.generator
        discriminator = model_.discriminator

        self.vocab_size = TINY_PARAMS.vocab_size
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(real_output, fake_output):
          real_loss = cross_entropy(tf.ones_like(real_output), real_output)
          fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
          total_loss = real_loss + fake_loss
          return total_loss

        def generator_loss(fake_output, generated_output, targets):
          discriminator_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
          logits = metrics.MetricLayer(self.vocab_size)([generated_output, targets])
          logits, generator_loss = metrics.LossLayer(self.vocab_size, 0.1)([logits, targets])
          return discriminator_loss + generator_loss, generator_loss




        for epoch in range(self.num_epochs):
            model_.train()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss_of_d = n_iter = total_loss_of_g = total_loss_of_original = 0




            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                    targets = Y
                    generated_output, fake_output = model_(inputs = X, targets = Y)
                    real_data = tf.one_hot(targets, TINY_PARAMS.vocab_size, dtype=tf.float32)
                    real_output = discriminator(real_data)

                    loss_of_g, loss_of_original = generator_loss(fake_output, generated_output, targets)
                    loss_of_d = discriminator_loss(real_output, fake_output)

                    grad_of_g = gen_tape.gradient(loss_of_g, generator.all_weights)
                    grad_of_d = disc_tape.gradient(loss_of_d, discriminator.all_weights)


                    optimizer.apply_gradients(zip(grad_of_g, generator.all_weights))
                    optimizer.apply_gradients(zip(grad_of_d, discriminator.all_weights))


                total_loss_of_g += loss_of_g
                total_loss_of_d += loss_of_d
                total_loss_of_original += loss_of_original
                n_iter += 1

            model_.eval()
            test_sample = trainX[0:2, :]
            model_.eval()
            prediction = generator(inputs = test_sample)
            print("Prediction: >>>>>  ", prediction["outputs"], "\n Target: >>>>>  ", trainY[0:2, :], "\n\n")

            print('Epoch [{}/{}]: loss_of_g {:.4f}, loss_of_d {:.4f}, loss_original {:.4f}'.format(epoch + 1, self.num_epochs, total_loss_of_g / n_iter, total_loss_of_d/n_iter, total_loss_of_original/n_iter))


if __name__ == '__main__':
    unittest.main()


#ljas
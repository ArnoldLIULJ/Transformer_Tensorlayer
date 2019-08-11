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
from seq2seq import Seq2seqLuongAttention
# from weightLightModels.transformer import Transformer
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
        cls.trainX = np.random.randint(low=2, high=50, size=(50, 11))
        cls.trainY = np.random.randint(low=2, high=50, size=(50, 10))

        cls.trainX[:,-1] = 1
        cls.trainY[:,0] = 0
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
        trace_path = "checkpoints_tl/logging/loss"
        vocabulary_size = 64
        emb_dim = 32
        model_ = Seq2seqLuongAttention(
            hidden_size=128,
            embedding_layer = tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim),
            cell=tf.keras.layers.GRUCell,
            method = "dot"
        )

        # print(", ".join(x for x in [t.name for t in model_.trainable_weights]))

        self.vocab_size = 64
        optimizer = tf.optimizers.Adam(learning_rate=0.01)
        for epoch in range(self.num_epochs):
            model_.train()
            t = time.time()
            trainX, trainY = shuffle(self.trainX, self.trainY)
            total_loss, n_iter = 0, 0
            for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=self.batch_size,
                                                    shuffle=False), total=self.n_step,
                             desc='Epoch[{}/{}]'.format(epoch + 1, self.num_epochs), leave=False):

                with tf.GradientTape() as tape:
                    dec_seq = Y[:, :-1]
                    targets = Y[:, 1:]
                    logits = model_(inputs = [X, dec_seq])
                    logits = metrics.MetricLayer(self.vocab_size)([logits, targets])
                    logits, loss = metrics.LossLayer(self.vocab_size, 0.1)([logits, targets])
                    
                    with tf.io.gfile.GFile(trace_path, "ab+") as trace_file:
                        trace_file.write(str(loss.numpy())+'\n')
                    grad = tape.gradient(loss, model_.all_weights)
                    optimizer.apply_gradients(zip(grad, model_.all_weights))
                    
            
                total_loss += loss
                n_iter += 1
            print(time.time()-t)
            # tl.files.save_npz(model_.all_weights, name='./model_v4.npz')
            model_.eval()
            test_sample = trainX[0:2, :]
            prediction = model_(inputs = [test_sample], seq_length = 10, sos=0)
            print("Prediction: >>>>>  ", prediction, "\n Target: >>>>>  ", trainY[0:2, 1:], "\n\n")

            print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, self.num_epochs, total_loss / n_iter))


if __name__ == '__main__':
    unittest.main()

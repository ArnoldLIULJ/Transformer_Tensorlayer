import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse
from utils.tokenizer import *
from models import model_params
from models.transformer_v2 import Transformer
from utils.pipeline_dataset import train_input_fn
from utils import metrics




def train_model(input_params):
    params = model_params.EXAMPLE_PARAMS
    dataset = train_input_fn(input_params)
    
    num_epochs = 50
    # @tf.function
    
    def train_step(inputs, targets):
        
        with tf.GradientTape() as tape:
            #print(inputs)
            
            logits = model(inputs=inputs, targets=targets)
            logits = metrics.MetricLayer(params.vocab_size)([logits, targets])
            logits, loss = metrics.LossLayer(params.vocab_size, 0.1)([logits, targets])

            
        gradients = tape.gradient(loss, model.all_weights)
        optimizer.apply_gradients(zip(gradients, model.all_weights))
        return loss

    
    model = Transformer(params)
    model.train()
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    
    
    for epoch in range(num_epochs):
        total_loss, n_iter = 0, 0
        for i, [inputs, targets] in enumerate(dataset):
            loss = train_step(inputs, targets)
            if (i % 100 == 0):
                print('Batch ID {} at Epoch [{}/{}]: loss {:.4f}'.format(i, epoch + 1, num_epochs, loss))
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
        tl.files.save_npz(model.all_weights, name='./checkouts_tl/model.npz')





if __name__ == '__main__':


    params = {}
    params["batch_size"] = 2048
    params["max_length"] = 256
    params["num_parallel_calls"] = 1
    params["repeat_dataset"] = 1
    params["static_batch"] = False
    params["num_gpus"] = 1
    params["use_synthetic_data"] = False
    params["data_dir"] = './data/data/wmt32k-train-00001*'
    train_model(params)

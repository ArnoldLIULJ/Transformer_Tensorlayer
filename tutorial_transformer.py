import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse
from utils.tokenizer import *
from models import model_params
from models.transformer import Transformer
from utils.pipeline_dataset import train_input_fn

def get_dataset():
    def _parse_example(serialized_example):
        """Return inputs and targets Tensors from a serialized tf.Example."""
        data_fields = {
            "inputs": tf.io.VarLenFeature(tf.int64),
            "targets": tf.io.VarLenFeature(tf.int64)
        }
        parsed = tf.io.parse_single_example(serialized_example, data_fields)
        inputs = tf.sparse.to_dense(parsed["inputs"])
        targets = tf.sparse.to_dense(parsed["targets"])
        return inputs, targets

    def _load_records(filename):
        """Read file and return a dataset of tf.Examples."""
        return tf.data.TFRecordDataset(filename, buffer_size=512)

    dataset = tf.data.Dataset.list_files('./data/data/wmt32k-train-00001*')
    dataset = dataset.interleave(_load_records, cycle_length=2)
    dataset = dataset.map(_parse_example)
    batch_size = 4096
    max_length = 4096

    dataset = dataset.padded_batch(batch_size=batch_size // max_length,
                                   padded_shapes=([max_length], [max_length]),
                                   drop_remainder=True)
    return dataset



def train_model(input_params):
    

    dataset = train_input_fn(input_params)
    print(dataset)
    num_epochs = 50
    # @tf.function
    def train_step(inputs, targets):
        model.train()
        with tf.GradientTape() as tape:
            #print(inputs)

            predictions = model(inputs=inputs, targets=targets)
            predictions = tf.reshape(predictions, [-1, predictions.shape[-1]])
            loss = tl.cost.cross_entropy_seq(target_seqs=targets, logits=predictions)
            
        gradients = tape.gradient(loss, model.all_weights)
        optimizer.apply_gradients(zip(gradients, model.all_weights))
        return loss

    params = model_params.EXAMPLE_PARAMS
    model = Transformer(params)
    print(model.name)
    print(model.all_layers)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    
    for epoch in range(num_epochs):
        total_loss, n_iter = 0, 0
        for i, [inputs, targets] in enumerate(dataset):
            loss = train_step(inputs, targets)
            loss = 0
            if (i % 100 == 0):
                print('Batch ID {} at Epoch [{}/{}]: loss {:.4f}'.format(i, epoch + 1, num_epochs, loss))
            total_loss += loss
            n_iter += 1

        # printing average loss after every epoch
        print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
        tl.files.save_npz(model.all_weights, name='model.npz')





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
    # wmt_dataset.download_and_preprocess_dataset('data/raw', 'data', search=False)
    train_model(params)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from utils.wmt_dataset import *
from utils.tokenizer import *
from models import model_params
from models.transformer import Transformer
from utils import wmt_dataset


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
    batch_size = 512
    max_length = 128

    dataset = dataset.padded_batch(batch_size=batch_size // max_length,
                                   padded_shapes=([max_length], [max_length]),
                                   drop_remainder=True)
    print ("=====", dataset)
    return dataset



class myModel(tl.models.Model):
    def __init__(self, params):
        super(myModel, self).__init__()
        self.transformer = Transformer(params)

    def forward(self, inputs, targets):
        return self.transformer(inputs=inputs, targets=targets)


def train_model(Subtokenizer):
    # @tf.function
    def train_step(inputs, targets):
        model.train()
        with tf.GradientTape() as tape:
            #print(inputs)

            predictions = model(inputs=inputs, targets=targets)
            predictions = tf.reshape(predictions, [-1, predictions.shape[-1]])
            loss = tl.cost.cross_entropy_seq(target_seqs=targets, logits=predictions)
            


        model.eval()
        print("inputs = ", inputs.shape)
        predictions = model(inputs=inputs)
        print("predictions = ", predictions)


        print(loss)
        gradients = tape.gradient(loss, model.all_weights)
        optimizer.apply_gradients(zip(gradients, model.all_weights))


    params = model_params.EXAMPLE_PARAMS
    #model = myModel(params)
    model = Transformer(params)
    # print(model.encoder_stack.weights)
    print(model.name)
    print(model.all_layers)

    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    dataset = get_dataset()
    dataset = dataset.repeat(5)

    for inputs, targets in dataset:
        train_step(inputs, targets)




def prepare_Subtokenizer():
        
        # Data sources for training/evaluating the transformer translation model.
    # If any of the training sources are changed, then either:
    #   1) use the flag `--search` to find the best min count or
    #   2) update the _TRAIN_DATA_MIN_COUNT constant.
    # min_count is the minimum number of times a token must appear in the data
    # before it is added to the vocabulary. "Best min count" refers to the value
    # that generates a vocabulary set that is closest in size to _TARGET_VOCAB_SIZE.
    _TRAIN_DATA_SOURCES = [
        {
            "url": "http://data.statmt.org/wmt17/translation-task/"
                "training-parallel-nc-v12.tgz",
            "input": "news-commentary-v12.de-en.en",
            "target": "news-commentary-v12.de-en.de",
        },
        {
            "url": "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
            "input": "commoncrawl.de-en.en",
            "target": "commoncrawl.de-en.de",
        },
        {
            "url": "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
            "input": "europarl-v7.de-en.en",
            "target": "europarl-v7.de-en.de",
        },
    ]
    # Use pre-defined minimum count to generate subtoken vocabulary.
    _TRAIN_DATA_MIN_COUNT = 6

    _EVAL_DATA_SOURCES = [
        {
            "url": "http://data.statmt.org/wmt17/translation-task/dev.tgz",
            "input": "newstest2013.en",
            "target": "newstest2013.de",
        }
    ]

    # Vocabulary constants
    _TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
    _TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
    VOCAB_FILE = "vocab.ende.%d" % _TARGET_VOCAB_SIZE

    # Strings to inclue in the generated files.
    _PREFIX = "wmt32k"
    _TRAIN_TAG = "train"
    _EVAL_TAG = "dev"  # Following WMT and Tensor2Tensor conventions, in which the
                    # evaluation datasets are tagged as "dev" for development.

    # Number of files to split train and evaluation data
    _TRAIN_SHARDS = 100
    _EVAL_SHARDS = 1

    def find_file(path, filename, max_depth=5):
        for root, dirs, files in os.walk(path):
            if filename in files:
                return os.path.join(root, filename)

            # Don't search past max_depth
            depth = root[len(path) + 1:].count(os.sep)
            if depth > max_depth:
                del dirs[:]  # Clear dirs
        return None

    def get_raw_existed_files(raw_dir, data_source):
        raw_files = {
            "inputs": [],
            "targets": [],
        }  # keys
        for d in data_source:
            input_file, target_file = find_files(
                raw_dir, d["input"], d["target"])
            raw_files["inputs"].append(input_file)
            raw_files["targets"].append(target_file)
        return raw_files

    def find_files(path, input_filename, target_filename):
        input_file = find_file(path, input_filename)
        target_file = find_file(path, target_filename)
        return input_file, target_file

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", "-dd", type=str, default="./data/data",
        help="[default: %(default)s] Directory for where the "
            "translate_ende_wmt32k dataset is saved.",
        metavar="<DD>")
    parser.add_argument(
        "--raw_dir", "-rd", type=str, default="./data/raw",
        help="[default: %(default)s] Path where the raw data will be downloaded "
            "and extracted.",
        metavar="<RD>")
    parser.add_argument(
        "--search", action="store_true",
        help="If set, use binary search to find the vocabulary set with size"
            "closest to the target size (%d)." % _TARGET_VOCAB_SIZE)

    FLAGS, unparsed = parser.parse_known_args()
    # main(sys.argv)
    make_dir(FLAGS.raw_dir)
    make_dir(FLAGS.data_dir)
    train_files = get_raw_existed_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
    eval_files = get_raw_existed_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)
    train_files_flat = train_files["inputs"] + train_files["targets"]
    vocab_file = os.path.join(FLAGS.data_dir, VOCAB_FILE)
    subtokenizer = Subtokenizer.init_from_files(
        vocab_file, train_files_flat, _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
        min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT)

    return subtokenizer

if __name__ == '__main__':



    Subtokenizer = prepare_Subtokenizer()


    # wmt_dataset.download_and_preprocess_dataset('data/raw', 'data', search=False)
    train_model(Subtokenizer)

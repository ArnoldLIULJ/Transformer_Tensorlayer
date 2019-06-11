from utils.wmt_dataset import *
from utils.tokenizer import *

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

if __name__ == "__main__":
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
  train_files = get_raw_files(FLAGS.raw_dir, _TRAIN_DATA_SOURCES)
  eval_files = get_raw_files(FLAGS.raw_dir, _EVAL_DATA_SOURCES)
  train_files_flat = train_files["inputs"] + train_files["targets"]
  vocab_file = os.path.join(FLAGS.data_dir, VOCAB_FILE)
  subtokenizer = Subtokenizer.init_from_files(
      vocab_file, train_files_flat, _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
      min_count=None if FLAGS.search else _TRAIN_DATA_MIN_COUNT)
  print(subtokenizer.decode([65,2150,100]))
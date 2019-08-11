
from utils.tokenizer import *
vocab_file = "vocab_mock"
file_name = "README.md"
_TRAIN_DATA_MIN_COUNT = 1
_TARGET_THRESHOLD = 1
_TARGET_VOCAB_SIZE = 10
subtokenizer = Subtokenizer.init_from_files(
    vocab_file, [file_name], _TARGET_VOCAB_SIZE, _TARGET_THRESHOLD,
    min_count=1)


print(subtokenizer.encode("I go university"))
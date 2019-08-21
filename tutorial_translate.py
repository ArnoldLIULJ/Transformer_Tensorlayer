# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Translate text or files using trained transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os


# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
import tensorflow as tf
import tensorlayer as tl
# pylint: enable=g-bad-import-order

from utils import tokenizer
from models import model_params
from weightLightModels import model_params as model_params_dw
from models.transformer_v2 import Transformer
from weightLightModels.transformer import Transformer as Transformer_DW
_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6


_TARGET_VOCAB_SIZE = 32768  # Number of subtokens in the vocabulary list.
_TARGET_THRESHOLD = 327  # Accept vocabulary if size is within this threshold
VOCAB_FILE = "vocab.ende.%d" % _TARGET_VOCAB_SIZE


def _get_sorted_inputs(filename):
  """Read and sort lines from the file sorted by decreasing length.

  Args:
    filename: String name of file to read inputs from.
  Returns:
    Sorted list of inputs, and dictionary mapping original index->sorted index
    of each element.
  """
  with tf.io.gfile.GFile(filename) as f:
    records = f.read().split("\n")
    inputs = [record.strip() for record in records]
    if not inputs[-1]:
      inputs.pop()

  input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
  sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

  sorted_inputs = [None] * len(sorted_input_lens)
  sorted_keys = [0] * len(sorted_input_lens)
  for i, (index, _) in enumerate(sorted_input_lens):
    sorted_inputs[i] = inputs[index]
    sorted_keys[index] = i

  return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):
  """Encode line with subtokenizer, and add EOS id to the end."""
  return subtokenizer.encode(line) + [tokenizer.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
  """Trim EOS and PAD tokens from ids, and decode to return a string."""
  try:
    
    index = list(ids.numpy()).index(tokenizer.EOS_ID)
    return subtokenizer.decode(ids[:index])
  except ValueError: 
    # print("mother fucker") # No EOS found in sequence
    return subtokenizer.decode(ids)


def translate_file(
    model, subtokenizer, input_file, output_file=None,
    print_all_translations=True):
  """Translate lines in file, and save to output file if specified.

  Args:
    model: tl.model
    subtokenizer: Subtokenizer object for encoding and decoding source and
       translated lines.
    input_file: file containing lines to translate
    output_file: file that stores the generated translations.
    print_all_translations: If true, all translations are printed to stdout.

  Raises:
    ValueError: if output file is invalid.
  """
  batch_size = _DECODE_BATCH_SIZE

  # Read and sort inputs by length. Keep dictionary (original index-->new index
  # in sorted list) to write translations in the original order.
  sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
  num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1

  def input_generator():
    """Yield encoded strings from sorted_inputs."""
    for i, line in enumerate(sorted_inputs):
      if i % batch_size == 0:
        batch_num = (i // batch_size) + 1
      yield _encode_and_add_eos(line, subtokenizer)

  def input_fn():
    """Created batched dataset of encoded inputs."""
    ds = tf.data.Dataset.from_generator(
        input_generator, tf.int64, tf.TensorShape([None]))
    ds = ds.padded_batch(batch_size, [None])
    return ds

  translations = []
  model.eval()
  for i, text in enumerate(input_fn()):
    prediction = model(inputs=text)
    for i, single in enumerate(prediction["outputs"]):
        translation = _trim_and_decode(single, subtokenizer)
        translations.append(translation)

  # Write translations in the order they appeared in the original file.
  if output_file is not None:
    if tf.io.gfile.isdir(output_file):
      raise ValueError("File output is a directory, will not save outputs to "
                       "file.")
    # tf.logging.info("Writing to file %s" % output_file)
    with tf.io.gfile.GFile(output_file, "w") as f:
      for i in sorted_keys:
        f.write("%s\n" % translations[i])



if __name__ == "__main__":
  subtokenizer = tokenizer.Subtokenizer("data/data/"+VOCAB_FILE)


  if (len(sys.argv) > 1 and sys.argv[1] == "tl"):
    params = model_params.EXAMPLE_PARAMS
    params.beam_size = 1
    model = Transformer(params)
    load_weights = tl.files.load_npz(name='./task/model.npz')
    tl.files.assign_weights(load_weights, model)
    input_file = "./data/data//newstest2014.en"
    translate_file(model, subtokenizer, input_file, output_file="./output/out_tl.de")


  if (len(sys.argv) > 1 and sys.argv[1] == "n-gram"):
    params = model_params.EXAMPLE_PARAMS
    params.beam_size = 1
    model = Transformer(params)
    load_weights = tl.files.load_npz(name='./checkpoints_v5/model.npz')
    tl.files.assign_weights(load_weights, model)
    input_file = "./data/data/newstest2014.en"
    translate_file(model, subtokenizer, input_file, output_file="./output/out_v5.de")



  


    

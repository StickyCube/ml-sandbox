import os
import urllib.request
import gzip
import numpy as np
from keras import utils

cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../.ml-sandbox-cache');

def load_file(url, filename, is_data):
  cache_filepath = os.path.join(cache_dir, filename)

  if not os.path.exists(cache_filepath):
    urllib.request.urlretrieve(url, cache_filepath)
  
  fd = gzip.GzipFile(cache_filepath, 'r')

  fd.read(4)

  n_items = int.from_bytes(fd.read(4), byteorder='big')

  if is_data:
    n_rows = int.from_bytes(fd.read(4), byteorder='big')
    n_cols = int.from_bytes(fd.read(4), byteorder='big')
    item_size = n_cols * n_rows
  else:
    item_size = 1

  data = np.zeros((n_items, item_size))

  for i in range(n_items):
    data[i, :] = np.frombuffer(fd.read(item_size), dtype=np.uint8)

  if is_data:
    return data / 255
  else:
    return utils.to_categorical(data)

def load():

  if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

  X = load_file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist-training-data.gz', True)
  Y = load_file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'mnist-training-labels.gz', False)
  X_test = load_file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist-test-data.gz', True)
  Y_test = load_file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'mnist-test-labels.gz', False)

  return X, Y, X_test, Y_test
  
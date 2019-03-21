import numpy as np
import sklearn.datasets
import keras.utils

np.random.seed(999)

def load():
  X, Y = sklearn.datasets.load_iris(return_X_y=True)

  # Take a random permutation of the data set
  permutation = np.random.permutation(np.arange(X.shape[0]))
  X = X[permutation]
  Y = Y[permutation]

  # convert the labels to a one-hot representation
  Y = keras.utils.to_categorical(Y)

  return X, Y

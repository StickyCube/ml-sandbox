import numpy as np
import sklearn.datasets
import keras.utils

np.random.seed(999)

def prediction_to_label(prediction):
  labels = ['setosa', 'versicolor', 'virginica']
  index = np.argmax(prediction)
  return labels[index]

def load():
  X, Y = sklearn.datasets.load_iris(return_X_y=True)

  m = X.shape[0]

  # Take a random permutation of the data set
  permutation = np.random.permutation(np.arange(m))

  X = X[permutation]
  Y = Y[permutation]

  # convert the labels to a one-hot representation
  Y = keras.utils.to_categorical(Y)
  
  m_train = int(0.8 * m)

  X_train = X[:m_train]
  X_test = X[m_train:]
  Y_train = Y[:m_train]
  Y_test = Y[m_train:]

  return X_train, Y_train, X_test, Y_test

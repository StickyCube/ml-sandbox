#! /usr/bin/env python3

"""Iris classification by neural network

The following model loads the iris dataset
and trains a network with 2 hidden layers
"""

from keras import layers, losses, optimizers, metrics, Sequential
from .datasets import iris, utils

def train_model(params = utils.DefaultParams()):
  X_train, Y_train, X_test, Y_test = iris.load()

  _, n_features = X_train.shape
  _, n_classes = Y_train.shape

  nn_layers = [
    layers.Dense(10, input_shape=(n_features,), activation='relu', name='dense-relu-1'),
    layers.Dense(8, activation='relu', name='dense-relu-2'),
    layers.Dense(n_classes, activation='softmax', name='output-layer')
  ]

  model = Sequential(nn_layers, name='iris_neural_network')

  optimizer = optimizers.Adam(lr=params.learning_rate, decay=params.learning_rate_decay)

  model.compile(
    optimizer,
    loss=losses.categorical_crossentropy,
    metrics=[metrics.categorical_accuracy]
  )

  model.summary()
  
  model.fit(X_train, Y_train, epochs=params.num_epochs, validation_data=(X_test, Y_test))

  return model

if __name__ == '__main__':
  train_model()
  

#! /usr/bin/env python3

from keras import layers, optimizers, losses, metrics, Sequential
from .datasets import mnist, utils

def train_model(params = utils.DefaultParams()):
  X_train, Y_train, X_test, Y_test = mnist.load(unrolled=True)

  _, n_features = X_train.shape
  _, n_classes = Y_train.shape

  nn_layers = [
    layers.Dense(100, activation='relu', input_shape=(n_features,), name='dense-relu-1'),
    layers.Dense(100, activation='relu', name='dense-relu-2'),
    layers.Dense(units=n_classes, activation='softmax', name='output-layer')
  ]

  model = Sequential(nn_layers, name='mnist_neural_network')

  optimizer = optimizers.Adam(lr=params.learning_rate, decay=params.learning_rate_decay)

  model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy]);
  
  model.summary()

  model.fit(x = X_train, y = Y_train, epochs=params.num_epochs, validation_data=(X_test, Y_test))

  return model

if __name__ == '__main__':
  train_model()

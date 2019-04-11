#! /usr/bin/env python3

"""
use a LeNet-5 style conv net with dropout to classify mnist handwritten digits
"""

from keras import layers, optimizers, losses, metrics, Sequential
from .datasets import mnist, utils

def train_model(params = utils.DefaultParams()):
  X_train, Y_train, X_test, Y_test = mnist.load(unrolled = False)
  
  _, *input_shape = X_train.shape
  _, n_classes = Y_train.shape

  print(X_train.shape)

  nn_layers = [
    layers.Conv2D(6, 5, strides=(1, 1), activation='relu', input_shape=input_shape, name='conv-1'),
    layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avg-pool-1'),
    layers.Dropout(0.1, name='dropout-1'),
    layers.Conv2D(16, 5, strides=(1, 1), activation='relu', name='conv-2'),
    layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='avg-pool-2'),
    layers.Dropout(0.1, name='dropout-2'),
    layers.Flatten(),
    layers.Dense(64, activation='relu', name='dense-relu-1'),
    layers.Dense(32, activation='relu', name='dense-relu-2'),
    layers.Dense(n_classes, activation='softmax', name='output-layer')
  ]

  model = Sequential(nn_layers, name='mnist_conv_net');

  optimizer = optimizers.Adam(lr = params.learning_rate, decay = params.learning_rate_decay)

  model.compile(optimizer, loss = losses.categorical_crossentropy, metrics = [metrics.categorical_accuracy])

  model.summary()

  model.fit(x = X_train, y = Y_train, validation_data=(X_test, Y_test), epochs = params.num_epochs)

  return model

if __name__ == '__main__':
  train_model()

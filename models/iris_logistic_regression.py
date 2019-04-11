#! /usr/bin/env python3

"""Iris classification by logistic regression

The following model loads the iris dataset from `sklean` and trains
a network with a single softmax layer - analagous to simple logistic regression
"""

from keras import layers, losses, optimizers, metrics, Sequential
from .datasets import iris, utils

def train_model(params = utils.DefaultParams()):
  X_train, Y_train, X_test, Y_test = iris.load()

  _, n_features = X_train.shape
  _, n_classes = Y_train.shape

  nn_layers = [
    layers.Dense(n_classes, input_shape=(n_features,), activation='softmax', name='output-layer')
  ]

  model = Sequential(nn_layers, name='iris_logistic_regression')

  optimizer = optimizers.Adam(lr=params.learning_rate, decay=params.learning_rate_decay)

  model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])

  model.summary()
  
  model.fit(
    X_train, 
    Y_train, 
    epochs=params.num_epochs, 
    validation_data=(X_test, Y_test)
  )

  return model;
  
if __name__ == '__main__':
  train_model()
  

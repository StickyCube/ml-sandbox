#! /usr/bin/env python3

"""Iris classification by neural network

The following model loads the iris dataset
a a network with 2 hidden layers

"""
from keras_tqdm import TQDMCallback
from keras import layers, regularizers, losses, optimizers, Sequential
import dataset

def train_model(X, Y, learning_rate, num_epochs, regularization_parameter):
  _, n_features = X.shape
  _, n_classes = Y.shape

  regularizer = regularizers.l2(regularization_parameter)

  model = Sequential([
    layers.Dense(10, 
      input_shape=(n_features,), 
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation='relu'),
    layers.Dense(8, 
      kernel_regularizer=regularizer,
      bias_regularizer=regularizer,
      activation='relu'),
    layers.Dense(n_classes, activation='softmax')
  ])

  optimizer = optimizers.Adam(lr=learning_rate)

  model.compile(optimizer,
      loss=losses.categorical_crossentropy,
      metrics=['accuracy'])

  model.summary()
  
  print('\n\n')

  history = model.fit(X, Y, 
    verbose=0, 
    callbacks=[TQDMCallback()], 
    epochs=num_epochs, 
    validation_split=0.2)
  
  print('\n\n')

  return model, history


def main(learning_rate, num_epochs, regularization_parameter):
  X, Y = dataset.load()
  
  train_model(X, Y, learning_rate, num_epochs, regularization_parameter)

if __name__ == '__main__':
  main(learning_rate=0.01, num_epochs=1000, regularization_parameter=0)
  

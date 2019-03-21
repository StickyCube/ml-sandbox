#! /usr/bin/env python3

"""Iris classification by logistic regression

The following model loads the iris dataset from `sklean` and trains
a a network with a single softmax layer - analagous to simple logistic regression

"""
from keras_tqdm import TQDMCallback
from keras import layers, losses, optimizers, Sequential
import dataset

def train_model(X, Y, learning_rate, num_epochs):
  _, n_features = X.shape
  _, n_classes = Y.shape

  model = Sequential([
    layers.Dense(n_classes, input_shape=(n_features,), activation='softmax')
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

def main(learning_rate, num_epochs):
  X, Y = dataset.load()
  
  train_model(X, Y, learning_rate, num_epochs)

if __name__ == '__main__':
  main(learning_rate=0.01, num_epochs=1000)
  

#! /usr/bin/env python3

from keras import layers, optimizers, losses, metrics, Sequential
import dataset

def train_model(params):
  X_train, Y_train, X_test, Y_test = dataset.load()

  _, n_features = X_train.shape
  _, n_classes = Y_train.shape

  model = Sequential([
    layers.Dense(units=params['layer_1_units'], activation='relu', input_shape=(n_features,)),
    layers.Dense(units=params['layer_2_units'], activation='relu'),
    layers.Dense(units=n_classes, activation='softmax')
  ])

  optimizer = optimizers.Adam(lr = params['learning_rate'], decay=params['decay'])

  model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy]);
  
  model.summary()

  h = model.fit(x = X_train, y = Y_train, epochs=params['num_epochs'], validation_data=(X_test, Y_test))

  return h.history

if __name__ == '__main__':
  parameters = {
    'layer_1_units': 100,
    'layer_2_units': 40,
    'learning_rate': 0.01,
    'decay': 0.01,
    'num_epochs': 100
  }

  train_model(parameters)

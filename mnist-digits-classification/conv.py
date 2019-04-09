#! /usr/bin/env python3

"""
use a LeNet-5 style conv net with dropout to classify mnist handwritten digits
"""

from keras import layers, optimizers, losses, metrics, Sequential
import dataset

def train_model(params):
  X_train, Y_train, X_test, Y_test = dataset.load(unrolled = False)
  
  _, *input_shape = X_train.shape
  _, n_classes = Y_train.shape

  print(X_train.shape)

  model = Sequential([
    layers.Conv2D(params['conv_1_filters'], params['conv_1_kernel_size'], strides = params['conv_1_strides'], activation='relu', input_shape = input_shape),
    layers.AveragePooling2D(pool_size = params['pool_1_kernel_size'], strides = params['pool_1_strides']),
    layers.Dropout(0.1),
    layers.Conv2D(params['conv_2_filters'], params['conv_2_kernel_size'], strides = params['conv_2_strides'], activation='relu'),
    layers.AveragePooling2D(pool_size = params['pool_2_kernel_size'], strides = params['pool_2_strides']),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(params['dense_1_units'], activation = 'relu'),
    layers.Dense(params['dense_2_units'], activation = 'relu'),
    layers.Dense(n_classes, activation = 'softmax')
  ]);

  optimizer = optimizers.Adam(lr = params['learning_rate'], decay = params['decay'])

  model.compile(optimizer, loss = losses.categorical_crossentropy, metrics = [metrics.categorical_accuracy])

  model.summary()

  model.fit(x = X_train, y = Y_train, validation_data=(X_test, Y_test), epochs = params['num_epochs'])

if __name__ == '__main__':
  parameters = {
    'conv_1_filters': 6,
    'conv_1_kernel_size': 5,
    'conv_1_strides': (1, 1),
    'pool_1_kernel_size': (2, 2),
    'pool_1_strides': (2, 2),
    'conv_2_filters': 16,
    'conv_2_kernel_size': 5,
    'conv_2_strides': (1, 1),
    'pool_2_kernel_size': (2, 2),
    'pool_2_strides': (2, 2),
    'dense_1_units': 64,
    'dense_2_units': 32,
    'learning_rate': 0.03,
    'decay': 0.01,
    'num_epochs': 50
  }

  train_model(parameters)

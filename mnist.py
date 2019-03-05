from tensorflow import keras
from keras.datasets import mnist
import numpy as np

import sansara as s

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train_cat = keras.utils.to_categorical(y_train)
y_test_cat = keras.utils.to_categorical(y_test)
print(np.shape(y_train_cat))

x_train = np.expand_dims(x_train, axis=-1) / 255.
x_test = np.expand_dims(x_test, axis=-1) / 255.
print(np.shape(x_train), np.min(x_train), np.max(x_train))


# lets try to outperform classic LeNet5 architecture
lenet5_dna = [['Conv2D', {'filters': 6, 'kernel_size': 5, 'padding': "'same'", 'use_bias': True}, -1],
              ['ReLU', {}, -1],
              ['AveragePooling2D', {'pool_size': 2}, -1],
              ['Conv2D', {'filters': 16, 'kernel_size': 5, 'padding': "'valid'", 'use_bias': True}, -1],
              ['ReLU', {}, -1],
              ['AveragePooling2D', {'pool_size': 2}, -1],
              ['Conv2D', {'filters': 120, 'kernel_size': 5, 'padding': "'valid'", 'use_bias': True}, -1],
              ['ReLU', {}, -1],
              ['Flatten', {}, -1],
              ['Dense', {'units': 84, 'activation': "'tanh'"}, -1],
              ['Dense', {'units': 10, 'activation': "'softmax'"}, -1]
             ]

input_shape = (28, 28, 1)
output_shape = (10,)

# assume 5 epochs will be enough to compare nets performance
# also you can change population_size (the number of nets training in parallel)
sansara_mnist = s.Sansara(input_shape, output_shape, population_size=50, base_dna=lenet5_dna, tag='mnist_lenet5', epochs=5)

# 50 generations takes a lot of time! feel free to decrease it.
sansara_mnist.main_cycle(50, x_train, y_train_cat, batch_size=32)


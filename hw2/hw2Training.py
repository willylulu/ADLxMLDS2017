import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten, Reshape, TimeDistributed
from keras.layers import recurrent
import keras.optimizers

x = np.load("x_data.npy")
y = np.load("x_label.npy")
y = keras.utils.to_categorical(y, num_classes=2573)
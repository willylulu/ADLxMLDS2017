import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Reshape
from keras.layers import recurrent
import keras.optimizers
import numpy as np
from sklearn.model_selection import train_test_split

X_train = np.load("../../data/mfcc_x.dat")
Y_train = np.load("../../data/y.dat")

print(Y_train)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.9)

print(x_train.shape)
print(y_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes=48)
y_test = keras.utils.to_categorical(y_test, num_classes=48)


data_dim = 39
timesteps = 1

# model = Sequential()
# model.add(Reshape((data_dim, timesteps), input_shape=(39,)))
# model.add(LSTM(48, return_sequences=True, activation='tanh'))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(48, return_sequences=True, activation='tanh'))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(48, return_sequences=True, activation='tanh'))  # return a single vector of dimension 32
# model.add(Flatten())
# model.add(Dense(48, activation='softmax'))

model = Sequential()
model.add(Dense(512, input_dim=39, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(48, activation='sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=128)
print(score)

model.save("mfcc_model.h5")
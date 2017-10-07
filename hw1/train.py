import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import recurrent
import keras.optimizers
import numpy as np
from sklearn.model_selection import train_test_split

X_train = np.load("../../data/mfcc_x.dat")
Y_train = np.load("../../data/y.dat")

print(Y_train)

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.8)

print(x_train.shape)
print(y_train.shape)

y_train = keras.utils.to_categorical(y_train, num_classes=48)
y_test = keras.utils.to_categorical(y_test, num_classes=48)

model = Sequential()
model.add(Dense(64, input_dim=39, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(48, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=8192)
print(score)
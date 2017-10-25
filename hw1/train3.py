import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Reshape, GRU, Bidirectional, SimpleRNN, Conv2D, MaxPooling2D, LeakyReLU, TimeDistributed, GaussianNoise
from keras.layers import recurrent
import keras.optimizers
import numpy as np
from sklearn.model_selection import train_test_split

X_train = np.load("../../data/mfcc_x.npy")
Y_train = np.load("../../data/y.npy")

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, train_size=0.9)

phoneClass = 39
z_train = np.zeros((y_train.shape[0], y_train.shape[1], 39))
z_test = np.zeros((y_test.shape[0], y_test.shape[1], 39))

for i in range(0,len(y_train)):
    for j in range(0, len(y_train[i])):
        temp = np.zeros(phoneClass)
        temp[int(y_train[i][j])] = 1.0
        z_train[i][j] = temp

for i in range(0,len(y_test)):
    for j in range(0, len(y_test[i])):
        temp = np.zeros(phoneClass)
        temp[int(y_test[i][j])] = 1.0
        z_test[i][j] = temp

timeStep = 123
data_dim = 39

model = Sequential()
# model.add(Conv1D(filters=256, kernel_size=3))
# model.add(Conv1D(filters=256, kernel_size=3))
# model.add(Conv1D(filters=256, kernel_size=3, activation= "relu"))
# model.add(Dropout(0.5)) 
# model.add(SimpleRNN(256, activation='relu', return_sequences=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal'))
# model.add(GRU(256, activation='relu', return_sequences=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
# model.add(Conv1D(filters=256, kernel_size=3))
# model.add(LeakyReLU())
# model.add(Dropout(0.5)) 
# model.add(SimpleRNN(256, activation='relu', return_sequences=True, kernel_initializer='orthogonal', recurrent_initializer='orthogonal'))
# model.add(GRU(256, activation='relu', return_sequences=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
# model.add(Conv1D(filters=256, kernel_size=3))
# model.add(LeakyReLU())
# model.add(Dropout(0.5))
# model.add(Bidirectional(SimpleRNN(512, activation='tanh', return_sequences=True, use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'), input_shape=(timeStep,data_dim)))
model.add(Reshape((timeStep,data_dim,1), input_shape=(timeStep,data_dim)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(Reshape((timeStep,data_dim*256)))
model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal')))
model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal')))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal')))
model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal')))
model.add(Dropout(0.5))
# model.add(GRU(512, activation='relu', return_sequences=True))
# model.add(GRU(512, activation='relu', return_sequences=True))
# model.add(GRU(256, activation='tanh', return_sequences=True, recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
# model.add(SimpleRNN(512, activation='relu', return_sequences=True, use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
# model.add(SimpleRNN(32, activation='relu', return_sequences=False, use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'))
# model.add(Dropout(0.5))  # returns a sequence of vectors of dimension 32  
# model.add(Dense(256, activation='relu'))
model.add(TimeDistributed(Dense(phoneClass, activation='softmax')))
# model.add(Reshape((timeStep,phoneClass), input_shape=(timeStep*data_dim,)))

# model = Sequential()
# model.add(Dense(512, input_dim=39, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(48, activation='sigmoid'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, z_train,
          epochs=100,
          batch_size=32,
          validation_data=(x_test, z_test),
          callbacks= [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')])
score = model.evaluate(x_test, z_test, batch_size=32)

model.save("mfcc_model_"+ str(score[1]) +".h5")

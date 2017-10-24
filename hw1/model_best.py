import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Flatten, Reshape, GRU, Bidirectional, SimpleRNN, Conv2D, MaxPooling2D, LeakyReLU, TimeDistributed, GaussianNoise
from keras.layers import recurrent
import keras.optimizers
import numpy as np
from sklearn.model_selection import train_test_split

X_train = np.load("../../data/mfcc_x.npy")
Y_train = np.load("../../data/y.npy")

phoneClass = 39
z_train = np.zeros((Y_train.shape[0], Y_train.shape[1], 39))

for i in range(0,len(Y_train)):
    for j in range(0, len(Y_train[i])):
        temp = np.zeros(phoneClass)
        temp[int(Y_train[i][j])] = 1.0
        z_train[i][j] = temp


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
model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal'), input_shape=(timeStep,data_dim)))
model.add(Bidirectional(LSTM(512, activation='tanh', return_sequences=True, kernel_initializer= 'glorot_uniform', recurrent_initializer='orthogonal')))
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

model.fit(X_train, z_train,epochs=12,batch_size=32)

model.save("mfcc_model_final.h5")

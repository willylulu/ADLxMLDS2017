import os
import sys
import numpy as np
from skimage import io, transform
import tensorflow as tf
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def dick(x):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=2)
    x = K.tile(x, [1, 4, 4, 1])
    return x

class dcgan():
    def __init__(self):
        self.dis_depth = 64
        self.gen_depth = 512
        self.alpha = 0.2
        self.dim = 4
        self.batch = 64
        self.epochs = 300
            
        self.generatorModel = load_model("pre-trained_model.h5", custom_objects={'dick':dick})
        
dcgan = dcgan()

lines = open(sys.argv[1], 'r').read().splitlines()

hair = {'orange':0, 'white':1, 'aqua':2, 'gray':3, 'green':4, 'red':5, 'purple':6, 'pink':7, 'blue':8, 'black':9, 'brown':10, 'blonde':11}
eyes = {'gray':12, 'black':13, 'orange':14, 'pink':15, 'yellow':16, 'aqua':17, 'purple':18, 'green':19, 'brown':20, 'red':21, 'blue':22}

noise = np.load('golden_noise_299.npy')

for x in lines:
    print(x)
    number = x.split(',')[0]
    string = x.split(',')[1]
    strs = string.split(' ')
    cond = np.zeros([5,23])
    cond[:, int(hair[strs[0]])] = 1
    cond[:, int(eyes[strs[2]])] = 1
    ite = 1
    images = dcgan.generatorModel.predict([noise, cond])
    
    for image in images:
        image = image/2+0.5
        io.imsave('samples/sample_' + number + '_' + str(ite) + '.jpg', image)
        ite += 1
    
import os
import random
import numpy as np
from skimage import io, transform
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Reshape, Input, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.engine.topology import Layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

K.set_learning_phase(False) 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

class InstanceNormalization2D(Layer):
    
    def __init__(self, **kwargs):
        super(InstanceNormalization2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[3],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[3],), initializer="zero", trainable=True)
        super(InstanceNormalization2D, self).build(input_shape)

    def call(self, x, mask=None):
        mean, variance = tf.nn.moments(x, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.shift

    def compute_output_shape(self, input_shape):
        return input_shape 

class gan():
        
    def __init__(self):
        self.d_dim = 64
        self.g_dim = 64
        self.alpha = 0.2
        self.imageshape = (64, 64, 3)

        self.truncate_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        self.random_normal = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)

        self.get_general_weight()
        self.get_generator_weight()
        self.get_discriminator_weight()
        self.get_generator_model()
        self.get_discriminator_model()


    def get_general_weight(self):
#             general
        self.lk = LeakyReLU(alpha=0.2)
        self.rl = Activation('relu')
        self.dp = Dropout(0.5)
        self.ft = Flatten()
        self.ct = Concatenate(axis=-1)
        self.tanh = Activation('tanh')

    def get_generator_weight(self):   
#             generator weight
        self.gc1 = Conv2D(self.g_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb1 = InstanceNormalization2D()
#             32 32 64
        self.gc2 = Conv2D(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb2 = InstanceNormalization2D()
#             16 16 128   
        self.gc3 = Conv2D(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb3 = InstanceNormalization2D()
#             8 8 256
        self.gc4 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb4 = InstanceNormalization2D()
#             4 4 512
        self.gc5 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb5 = InstanceNormalization2D()
#             2 2 512

        self.gc6 = Conv2D(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb6 = InstanceNormalization2D()
#             1 1 512

        self.gc7 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb7 = InstanceNormalization2D()
#             2 2 512
        self.gc8 = Conv2DTranspose(self.g_dim*8, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb8 = InstanceNormalization2D()
#             4 4 512
        self.gc9 = Conv2DTranspose(self.g_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb9 = InstanceNormalization2D()
#             8 8 256
        self.gc10 = Conv2DTranspose(self.g_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb10 = InstanceNormalization2D()
#             16 16 128   
        self.gc11 = Conv2DTranspose(self.g_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
        self.gb11 = InstanceNormalization2D()
#             32 32 64
        self.gc12 = Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)

    def get_discriminator_weight(self):
#             discriminator weight
        self.dc1 = Conv2D(self.d_dim, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             32 32 64
        self.dc2 = Conv2D(self.d_dim*2, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             16 16 128
        self.dc3 = Conv2D(self.d_dim*4, 4, strides=2, padding='same', kernel_initializer=self.truncate_normal)
#             8 8 256
        self.dc4 = Conv2D(self.d_dim*8, 4, strides=1, padding='same', kernel_initializer=self.truncate_normal)
#             4 4 512
        self.dc5 = Conv2D(1, 1, strides=1, padding='same', kernel_initializer=self.truncate_normal)
        self.dd1 = Dense(1, kernel_initializer=self.truncate_normal)

    def get_generator_model(self):   
#             generator model
        self.ginput = Input(shape=self.imageshape)
        self.ng1 = self.gb1(self.gc1(self.ginput))
        self.ng2 = self.gb2(self.gc2(self.lk(self.ng1)))
        self.ng3 = self.gb3(self.gc3(self.lk(self.ng2)))
        self.ng4 = self.gb4(self.gc4(self.lk(self.ng3)))
        self.ng5 = self.gb5(self.gc5(self.lk(self.ng4)))

        self.ng6 = self.gb6(self.gc6(self.lk(self.ng5)))

        self.ng7 = self.dp(self.gc7(self.rl(self.ng6)))
        self.ng8 = self.ct([self.gb7(self.ng7), self.ng5])

        self.ng9 = self.dp(self.gc8(self.rl(self.ng8)))
        self.ng10 = self.ct([self.gb8(self.ng9), self.ng4])

        self.ng11 = self.dp(self.gc9(self.rl(self.ng10)))
        self.ng12 = self.ct([self.gb9(self.ng11), self.ng3])  

        self.ng13 = self.dp(self.gc10(self.rl(self.ng12)))
        self.ng14 = self.ct([self.gb10(self.ng13), self.ng2])  

        self.ng15 = self.dp(self.gc11(self.rl(self.ng14)))
        self.ng16 = self.ct([self.gb11(self.ng15), self.ng1])

        self.ng17 = self.tanh(self.gc12(self.rl(self.ng16)))

        self.generator = Model(inputs=[self.ginput], outputs=[self.ng17])

    def get_discriminator_model(self):  
#             discriminator model
        self.dinput = Input(shape=self.imageshape)
        self.nd1 = self.lk(self.dc1(self.dinput))
        self.nd2 = self.lk(self.dc2(self.nd1))
        self.nd3 = self.lk(self.dc3(self.nd2))
        self.nd4 = self.lk(self.dc4(self.nd3))
        self.nd5 = self.dc5(self.nd4)
#         self.nd5 = self.dd1(self.ft(self.nd4))

        self.discriminator = Model(inputs=[self.dinput], outputs=[self.nd5])
        
    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

class CycleGAN():
    
    def __init__(self):
        self.batch = 1
        self.epochs = 300
        self.imageshape = (64, 64, 3)
        
        self.lr=2e-4
        
        self.x2y = gan()
        self.y2x = gan()
        self.x2y.summary()
        self.y2x.summary()
        self.trainingModel()
        
    def trainingModel(self):
        
        self.real_x = Input(shape=self.imageshape) #anime
        self.real_y = Input(shape=self.imageshape) #face
        
        self.fake_x = self.y2x.generator([self.real_y]) #fake anime
        self.fake_y = self.x2y.generator([self.real_x]) #fake face
        
        self.fake_xx = self.y2x.generator([self.fake_y]) #cycle anime
        self.fake_yy = self.x2y.generator([self.fake_x]) #cycle face
        
        self.d_fake_x = self.y2x.discriminator([self.fake_x])
        self.d_fake_y = self.x2y.discriminator([self.fake_y])
        
        self.d_real_x = self.y2x.discriminator([self.real_x])
        self.d_real_y = self.x2y.discriminator([self.real_y])
        
        self.g_loss1 = K.mean(K.square(K.ones_like(self.d_fake_x) - K.sigmoid(self.d_fake_x)), axis=-1)
        self.g_loss2 = K.mean(K.square(K.ones_like(self.d_fake_y) - K.sigmoid(self.d_fake_y)), axis=-1)
        self.g_loss3 = K.mean(K.abs(self.real_x - self.fake_xx))
        self.g_loss4 = K.mean(K.abs(self.real_y - self.fake_yy))
        self.g_loss = self.g_loss1 + self.g_loss2 + 10 * self.g_loss3 + 10 * self.g_loss4
        
        self.d_loss1 = K.mean(K.square(K.ones_like(self.d_real_x) - K.sigmoid(self.d_real_x)), axis=-1)
        self.d_loss2 = K.mean(K.square(K.zeros_like(self.d_fake_x) - K.sigmoid(self.d_fake_x)), axis=-1)
        self.d_loss3 = K.mean(K.square(K.ones_like(self.d_real_y) - K.sigmoid(self.d_real_y)), axis=-1)
        self.d_loss4 = K.mean(K.square(K.zeros_like(self.d_fake_y) - K.sigmoid(self.d_fake_y)), axis=-1)
        self.d_loss = (self.d_loss1 + self.d_loss2 + self.d_loss3 + self.d_loss4)/2
    
    def set_lr(self, lr):
        
        self.d_training_updates = Adam(lr=lr, beta_1=0.5).get_updates(self.x2y.discriminator.trainable_weights + self.y2x.discriminator.trainable_weights,[], self.d_loss)
        self.d_train = K.function([self.real_x, self.real_y], [self.d_loss], self.d_training_updates)
        
        self.g_training_updates = Adam(lr=lr, beta_1=0.5).get_updates(self.x2y.generator.trainable_weights + self.y2x.generator.trainable_weights,[], self.g_loss)
        self.g_train = K.function([self.real_x, self.real_y], [self.g_loss], self.g_training_updates)
        
    def train(self, datasetx, datasety):
        
        print("training...")
        self.set_lr(self.lr)
        for k in range(0, self.epochs):
            
            ite=0
            while ite<len(datasetx) and ite<len(datasety):
                datax = datasetx[ite:(ite+self.batch)]
                datay = datasety[ite:(ite+self.batch)]
                
                for l in range(1):
                    errD, = self.d_train([datax, datay])
                    errD = np.mean(errD)
                for l in range(1):
                    errG, = self.g_train([datax, datay])
                    errG = np.mean(errG)
                print(errD, errG)
                ite+=self.batch
                
                if ite%50==0 and ite>0:
                    print("save")
                    pseed = np.random.randint(len(datasetx), size=16)
                    imagea = datasetx[pseed]
                    imageb = datasety[pseed]
                    fakey = self.x2y.generator.predict([imagea])
                    fakex = self.y2x.generator.predict([imageb])
                    fakeyy = self.x2y.generator.predict([fakex])
                    fakexx = self.y2x.generator.predict([fakey])
                    images = np.concatenate([imagea[:8], fakexx[:8], fakey[:8], imageb[:8], fakeyy[:8], fakex[:8], imagea[8:], fakexx[8:], fakey[8:], imageb[8:], fakeyy[8:], fakex[8:]])
                    width = 8
                    height = 12
                    new_im = Image.new('RGB', (64*height,64*width))
                    for ii in range(height):
                        for jj in range(width):
                            index=ii*width+jj
                            image = (images[index]/2+0.5)*255
                            image = image.astype(np.uint8)
                            new_im.paste(Image.fromarray(image,"RGB"), (64*ii,64*jj))
                    filename = "image_cycle_gan/fakeFace%d.png"%k
                    new_im.save(filename)
                    self.x2y.generator.save("model_cycle_gan/generator_x2y%d.h5"%k)
                    self.y2x.generator.save("model_cycle_gan/generator_y2x%d.h5"%k)
                if ite%1000==0 and ite>0:
                    if self.lr > self.min_lr:
                        self.lr -= self.min_lr
                        
print("load anime...")
anime = np.load('data2.npy')
print(anime.shape)
print("load celeb...")
animegray = 0.299*anime[:,:,:,0] + 0.587*anime[:,:,:,1] + 0.114*anime[:,:,:,2]
animegray = np.expand_dims(animegray, axis=-1)
animegray = np.tile(animegray, [1, 1, 1, 3])
print(animegray.shape)
print("finish")

cyclegan = CycleGAN()
cyclegan.train(anime, animegray)
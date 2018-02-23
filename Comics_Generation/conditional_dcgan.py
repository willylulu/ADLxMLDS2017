import os
import random
import numpy as np
from skimage import io, transform
from PIL import Image
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, LeakyReLU, Activation, Reshape, Input
from keras.layers import Conv2D, Conv2DTranspose, Lambda, Concatenate
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

K.set_learning_phase(False) 
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
        
    def getmodelweight(self):
    
        #generator weights
        
        truncate_normal = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
        
        self.s_dense = Dense(256, kernel_initializer=truncate_normal, name='sd0')
        self.s_leak = LeakyReLU(self.alpha, name='slk')
        
        #4 4 512
        self.g_dense_1 = Dense(self.dim*self.dim*self.gen_depth, kernel_initializer=truncate_normal, name='gd1')
        self.g_conv_1 = Conv2DTranspose(int(self.gen_depth/2), 5, kernel_initializer=truncate_normal, strides=2, padding='same', name='gc1')
        self.g_conv_2 = Conv2DTranspose(int(self.gen_depth/4), 5, kernel_initializer=truncate_normal, strides=2, padding='same', name='gc2')
        self.g_conv_3 = Conv2DTranspose(int(self.gen_depth/8), 5, kernel_initializer=truncate_normal, strides=2, padding='same', name='gc3')
        self.g_conv_4 = Conv2DTranspose(3, 5, kernel_initializer=truncate_normal, strides=2, padding='same', name='gc4')
        
        self.g_batch_1 = BatchNormalization(momentum=0.9, epsilon=1e-5, name='gb1')
        self.g_batch_2 = BatchNormalization(momentum=0.9, epsilon=1e-5, name='gb2')
        self.g_batch_3 = BatchNormalization(momentum=0.9, epsilon=1e-5, name='gb3')
        self.g_batch_4 = BatchNormalization(momentum=0.9, epsilon=1e-5, name='gb4')
        
        self.g_resap = Reshape((self.dim, self.dim, self.gen_depth), name='gres')
        self.g_relu = Activation('relu', name='greu')
        self.g_leak = LeakyReLU(self.alpha, name='glk')
        self.g_tanh = Activation('tanh', name='gtan')
        self.g_concat = Concatenate(axis=-1, name='gcon')
        
        #discriminator weights
    
        #32 32 64
        self.d_conv_1 = Conv2D(self.dis_depth*1, 5, strides=2, kernel_initializer=truncate_normal, padding='same', name='dc1') 
        self.d_conv_2 = Conv2D(self.dis_depth*2, 5, strides=2, kernel_initializer=truncate_normal, padding='same', name='dc2')
        self.d_conv_3 = Conv2D(self.dis_depth*4, 5, strides=2, kernel_initializer=truncate_normal, padding='same', name='dc3')
        self.d_conv_4 = Conv2D(self.dis_depth*8, 5, strides=2, kernel_initializer=truncate_normal, padding='same', name='dc4')
        self.d_conv_5 = Conv2D(self.dis_depth*8, 1, strides=1, kernel_initializer=truncate_normal, padding='same', name='dc5')
        
        self.d_leak = LeakyReLU(self.alpha, name='dlk')
        self.d_lamb = Lambda(dick, name='dlm')
        self.d_concat = Concatenate(axis=-1, name='dc')
        self.d_flat = Flatten()
        self.d_dense = Dense(1, kernel_initializer=truncate_normal)
        self.d_sig = Activation('sigmoid')
        
    def getgenerator(self, input_1, input_2):
        
        #generator network
        
        self.gn_concat = self.g_concat([input_1, input_2])
        
        self.gn_dense_1 = self.g_dense_1(self.gn_concat)
        self.gn_batch_1 = self.g_batch_1(self.gn_dense_1)
        self.gn_resap_1 = self.g_resap(self.gn_batch_1)
        
        self.gn_conv_1 = self.g_conv_1(self.gn_resap_1)
        self.gn_batch_2 = self.g_batch_2(self.gn_conv_1)
        self.gn_relu_1 = self.g_relu(self.gn_batch_2)
        
        self.gn_conv_2 = self.g_conv_2(self.gn_relu_1)
        self.gn_batch_3 = self.g_batch_3(self.gn_conv_2)
        self.gn_relu_2 = self.g_relu(self.gn_batch_3)
        
        self.gn_conv_3 = self.g_conv_3(self.gn_relu_2)
        self.gn_batch_4 = self.g_batch_4(self.gn_conv_3)
        self.gn_relu_3 = self.g_relu(self.gn_batch_4)
        
        self.gn_conv_4 = self.g_conv_4(self.gn_relu_3)
        self.gn_tanh = self.g_tanh(self.gn_conv_4)
        
        return self.gn_tanh
        
    def getdiscriminator(self, input_1, input_2):
        
        #discriminator network
        
        #32 32 64
        self.dn_conv_1 = self.d_conv_1(input_1)
        self.dn_leak_1 = self.d_leak(self.dn_conv_1)
        
        #16 16 128
        self.dn_conv_2 = self.d_conv_2(self.dn_leak_1)
        self.dn_leak_2 = self.d_leak(self.dn_conv_2)
        
        #8 8 256
        self.dn_conv_3 = self.d_conv_3(self.dn_leak_2)
        self.dn_leak_3 = self.d_leak(self.dn_conv_3)
        
        #4 4 512
        self.dn_conv_4 = self.d_conv_4(self.dn_leak_3)
        self.dn_leak_4 = self.d_leak(self.dn_conv_4)
        
        self.dn_lamba = self.d_lamb(input_2)
        self.dn_concat = self.d_concat([self.dn_leak_4, self.dn_lamba])
        
        self.dn_conv_5 = self.d_conv_5(self.dn_concat)
        self.dn_leak_5 = self.d_leak(self.dn_conv_5)
        
        self.dn_flat = self.d_flat(self.dn_leak_5)
        
        self.dn_dense = self.d_dense(self.dn_flat)
        return self.dn_dense
        
    def getmodels(self):
        
        input_shape1 = (64, 64, 3)
        input_shape2 = (23,)
        input_shape3 = (100,)

        self.input_1 = Input(shape=input_shape1) #image
        self.input_2 = Input(shape=input_shape2) #seq vec
        self.input_3 = Input(shape=input_shape3) #random noise
        
        self.generator = self.getgenerator(self.input_3, self.input_2)
        self.discriminator = self.getdiscriminator(self.input_1, self.input_2)
        
        self.generatorModel = Model(inputs=[self.input_3, self.input_2], outputs=[self.generator])
        self.discriminatorModel = Model(inputs=[self.input_1, self.input_2], outputs=[self.discriminator])
        
        self.generatorModel.summary()
        self.discriminatorModel.summary()
        
        self.netD_real_input = Input(shape=(64, 64, 3))
        self.netD_wrong_input = Input(shape=(64, 64, 3))
        self.seq2vec = Input(shape=(23,))
        self.wrong_seq2vec = Input(shape=(23,))
        self.noisev = Input(shape=(100,))
        
        self.netD_fake_input = self.generatorModel([self.noisev, self.seq2vec])

#         self.epsilon_input = K.placeholder(shape=(None,1,1,1))
#         self.epsilont_input = K.placeholder(shape=(None,1))
#         self.netD_mixed_input = Input(shape=(64, 64, 3),
#             tensor=self.epsilon_input * self.netD_real_input + (1-self.epsilon_input) * (self.netD_fake_input+self.netD_wrong_input + self.netD_real_input)/3)
        
#         self.mixed_seq2vec = Input(shape=(2400,),
#             tensor=self.epsilont_input * self.seq2vec + (1-self.epsilont_input) * (self.seq2vec + self.seq2vec + self.wrong_seq2vec)/3)
        
        self.d_real = self.discriminatorModel([self.netD_real_input, self.seq2vec])
        self.d_fake = self.discriminatorModel([self.netD_fake_input, self.seq2vec])
        self.d_wrong_image = self.discriminatorModel([self.netD_wrong_input, self.seq2vec])
        self.d_wrong_tag = self.discriminatorModel([self.netD_real_input, self.wrong_seq2vec])
        
        self.d_loss1 = K.mean(K.binary_crossentropy(K.ones_like(self.d_real), K.sigmoid(self.d_real)), axis=-1)
        self.d_loss2 = K.mean(K.binary_crossentropy(K.zeros_like(self.d_fake), K.sigmoid(self.d_fake)), axis=-1)
        self.d_loss3 = K.mean(K.binary_crossentropy(K.zeros_like(self.d_wrong_image), K.sigmoid(self.d_wrong_image)), axis=-1)
        self.d_loss4 = K.mean(K.binary_crossentropy(K.zeros_like(self.d_wrong_tag), K.sigmoid(self.d_wrong_tag)), axis=-1)
#         self.grad_mixed = K.gradients(self.discriminatorModel([self.netD_mixed_input, self.mixed_seq2vec]), [self.netD_mixed_input])[0]
#         self.norm_grad_mixed = K.sqrt(K.sum(K.square(self.grad_mixed), axis=[1,2,3]))
#         self.grad_penalty = K.mean(K.square(self.norm_grad_mixed -1))

        self.d_loss = self.d_loss1 + self.d_loss2 + self.d_loss3 + self.d_loss4


        self.d_training_updates = Adam(lr=5e-5, beta_1=0.5, beta_2=0.9).get_updates(self.discriminatorModel.trainable_weights,[], self.d_loss)
        self.netD_train = K.function([self.netD_real_input, self.netD_wrong_input, self.seq2vec, self.wrong_seq2vec, self.noisev],
                                [self.d_loss],    
                                self.d_training_updates)
        
        self.g_loss = K.mean(K.binary_crossentropy(K.ones_like(self.d_fake), K.sigmoid(self.d_fake)), axis=-1)
        
        self.g_training_updates = Adam(lr=5e-5, beta_1=0.5, beta_2=0.9).get_updates(self.generatorModel.trainable_weights,[], self.g_loss)
        self.netG_train = K.function([self.noisev, self.seq2vec], [self.g_loss], self.g_training_updates)
        
    def loadseq2vec(self):
        self.seq2vec = np.load('condition.npy')
        
    def train(self):
        
        hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
        eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']
        aa = np.zeros([36,23])
        aa[:,hair.index('gray')]=1
        aa[:,len(hair)+eyes.index('gray')]=1

        # replace path to your numpy file, is a images pickle which shape is (img num, 64, 64, 3)
        bigdata = np.load('../../../GAN/data2.npy') 
        
        for k in range(0, self.epochs):
            
            ite=0
            while ite<len(bigdata):
                data = bigdata[ite:(ite+self.batch)]
                vecs = self.seq2vec[ite:(ite+self.batch)]
                wdata = [data[(x+1)%len(data)] for x in range(len(data))]
                wvecs = np.zeros([len(vecs), 23])
                    
                for l in range(len(vecs)):
                    a = random.randint(0, 11)
                    b = random.randint(12, 22)
                    wvecs[l][a] = 1
                    wvecs[l][b] = 1
                
                noise = np.random.normal(0, 1.0, size=[len(data), 100])
                
                for l in range(1):
                    errD,  = self.netD_train([data, wdata, vecs, wvecs, noise])
                    errD = np.mean(errD)

                for l in range(2):
                    errG, = self.netG_train([noise, vecs])
                    errG = np.mean(errG)

                print(errD, errG)
                ite+=self.batch
                    
            print("save")
            noise = np.random.normal(0, 1.0, size=[36, 100])
            images_fake = self.generatorModel.predict([noise, aa])
            width = 6
            new_im = Image.new('RGB', (64*width,64*width))
            for ii in range(width):
                for jj in range(width):
                    index=ii*width+jj
                    image = (images_fake[index]/2+0.5)*255
                    image = image.astype(np.uint8)
                    new_im.paste(Image.fromarray(image,"RGB"), (64*ii,64*jj))
            filename = "fakeFace%d.png"%k
            new_im.save(filename)
            self.generatorModel.save("generator%d.h5"%k)
            self.discriminatorModel.save("discriminator%d.h5"%k)
            
dcgan = dcgan()
dcgan.getmodelweight()
dcgan.getmodels()
dcgan.loadseq2vec()
dcgan.train()
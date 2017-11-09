import os
import numpy as np
import json
import string
from random import randint

import keras
import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper, ResidualWrapper

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest

from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops import beam_search_decoder

path = "../../data/MLDS_hw2_data/"
traindir = "training_data/feat/"
testdir = "testing_data/feat/"
trainfiles = os.listdir(path+traindir)
testfiles = os.listdir(path+testdir)

traindata = {}
for i in range(0,len(trainfiles)):
    traindata[str.split(trainfiles[i],".")[0]+'.'+str.split(trainfiles[i],".")[1]] = np.load(path+traindir+trainfiles[i])
print(len(traindata))

testdata = {}
for i in range(0,len(testfiles)):
    testdata[str.split(testfiles[i],".")[0]+'.'+str.split(testfiles[i],".")[1]] = np.load(path+testdir+testfiles[i])
print(len(testdata))

trainjsonfile = open(path+"training_label.json","r")
testjsonfile = open(path+"testing_label.json","r")
trainjson = json.load(trainjsonfile)
testjson = json.load(testjsonfile)

words = []
maxlen = -1
for x in trainjson:
    for y in x['caption']:
        y = ''.join(c for c in y if c not in string.punctuation)
        ss = unicode.split(y," ")
        if len(ss)>maxlen:
            maxlen = len(ss)
            maxlenStr = y
        for z in ss:
            words.append(z.lower())
print(maxlen)
print(maxlenStr)

encodeWords = {}
counter = 1
for x in words:
    if x not in encodeWords:
        encodeWords[x] = counter
        counter = counter + 1
encodeWords["<EOS>"] = 0
print(len(encodeWords))
words = []

decodeWords = {}
for key, value in encodeWords.items():
    decodeWords[value] = key
print(len(decodeWords))

def getMiniDataSets():
    x_data = np.zeros((1450,80,4096),dtype="float32")
    x_label = np.zeros((1450,40,1),dtype="float32")
    y_length = np.zeros((1450),dtype="int32")

    i = 0
    seqnum = 0
    for x in trainjson:
        name = x["id"]
        temp = traindata[name]
        counter2 = 0
        
        random = randint(0, len(x["caption"])-1)
        
        y = x["caption"][random]
        seqnum = seqnum + 1

        x_data[i] = temp

        y_length[i] = len(unicode.split(y," "))

        tempB = []
        y = ''.join(c for c in y if c not in string.punctuation)
        for z in unicode.split(y," "):
            tempB.append(encodeWords[z.lower()])

        y_length[i] = len(unicode.split(y," "))

        for xa in range(len(tempB),40):
            tempB.append(encodeWords["<EOS>"])
        tempB = np.reshape(tempB,(40,1))
        x_label[i] = tempB
        i = i+1
        
    print(seqnum)

    x_data = np.resize(x_data,(23,64,80,4096))
    x_label = np.resize(x_label,(23,64,40))    
    y_length = np.resize(y_length,(23,64))
    return x_data, x_label, y_length

#tensorflow   

inputs = tf.placeholder(tf.float32,[None,80,4096]) 
labels = tf.placeholder(tf.int32,[None,40])
length = tf.placeholder(tf.int32,[None])

sequence_length = [80 for _ in range(64)]

encoder_cell = tf.nn.rnn_cell.LSTMCell(256)

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs, dtype=tf.float32)
print(encoder_outputs.get_shape())

attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=256, memory=encoder_outputs)

decoder_cell = tf.contrib.rnn.BasicLSTMCell(256)
attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=64)
initial_state = initial_state.clone(cell_state=encoder_state)

projection_layer = Dense(6058)  

helper = tf.contrib.seq2seq.TrainingHelper(inputs=encoder_outputs, sequence_length=sequence_length)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
logits = tf.contrib.seq2seq.dynamic_decode(decoder,maximum_iterations=40)

logits = tf.nn.softmax(logits[0].rnn_output)

masks = tf.cast(tf.sequence_mask(length, maxlen=40),tf.float32);
loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=labels, weights=masks)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
minimize = optimizer.minimize(loss)
    
# #train_loss = (tf.reduce_sum(crossent * masks) / batch_size)


# opt = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(train_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for j in range(200):
    x_data, x_label, y_length = getMiniDataSets()
    
    print("")
    for i in range(23):
        outs = sess.run([minimize, loss], feed_dict={inputs: x_data[i], labels: x_label[i], length: y_length[i]})
        
    print(outs[1])

    prediction=tf.argmax(logits,-1)
    best = sess.run([prediction],feed_dict={inputs: x_data[randint(0, 22)]})
    best = np.reshape(best,(64,40))
    ram = [randint(0, 63)]
    print("{0} {1}".format( ram,best[ram]))
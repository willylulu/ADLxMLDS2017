import os
import numpy as np
import json
import string
from random import randint

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

trainfiles = os.listdir(path+traindir)

traindata = {}
for i in range(0,len(trainfiles)):
    traindata[str.split(trainfiles[i],".")[0]+'.'+str.split(trainfiles[i],".")[1]] = np.load(path+traindir+trainfiles[i])
print(len(traindata))

trainjsonfile = open(path+"training_label.json","r")

trainjson = json.load(trainjsonfile)


words = []
maxlen = -1
for x in trainjson:
    for y in x['caption']:
        y = ''.join(c for c in y if c not in string.punctuation)
        ss = str.split(y," ")
        if len(ss)>maxlen:
            maxlen = len(ss)
            maxlenStr = y
        for z in ss:
            words.append(z.lower())
encodeWords = {}
counter = 4

for x in words:
    if x not in encodeWords:
        encodeWords[x] = counter
        counter = counter + 1 
encodeWords["<PAD>"] = 0      
encodeWords["<BOS>"] = 1
encodeWords["<EOS>"] = 2
encodeWords["<NAN>"] = 3
print(len(encodeWords))

decodeWords = {}
for key, value in encodeWords.items():
    decodeWords[value] = key
print(len(decodeWords))

np.save("encodeWords.npy",encodeWords)
np.save("decodeWords.npy",decodeWords)

max_seq_length = 21

def getStr(ints):
    sentence = ' '.join([decodeWords[int] for int in ints])
    sentence = sentence.replace('<BOS> ','').replace(' <EOS>', '')
    return sentence

def getMiniDataSets():
    x_data = np.zeros((1450,80,4096),dtype="float32")
    x_label = np.zeros((1450,max_seq_length),dtype="int32")
    x_label_train = np.zeros((1450,max_seq_length),dtype="int32")
    y_length = np.zeros((1450),dtype="int32")
    y_length_train = np.zeros((1450),dtype="int32")

    i = 0
    for x in trainjson:
        name = x["id"]
        temp = traindata[name]
        counter2 = 0
        
        random = randint(0, len(x["caption"])-1)
        
        y = x["caption"][random]

        x_data[i] = temp

        x_label_temp = []
        x_label_train_temp = []
        
        y = ''.join(c for c in y if c not in string.punctuation)
        temp = [encodeWords[x.lower()] for x in str.split(y," ")]
        
        if(len(temp)>20):
            temp = temp[:20]
        
        x_label_temp = temp + [encodeWords["<EOS>"]]
        x_label_train_temp = [encodeWords["<BOS>"]] + temp 
        
        y_length[i] = len(x_label_temp)
        y_length_train[i] = len(x_label_train_temp) 

        for xa in range(len(x_label_temp),max_seq_length):
            x_label_temp.append(encodeWords["<PAD>"])
        for xa in range(len(x_label_train_temp),max_seq_length):
            x_label_train_temp.append(encodeWords["<PAD>"])                       
            
        x_label_temp = np.reshape(x_label_temp,(max_seq_length))
        x_label_train_temp = np.reshape(x_label_train_temp,(max_seq_length))
        
        x_label[i] = x_label_temp
        x_label_train[i] = x_label_train_temp 
        i = i+1
        
    x_data = np.split(x_data,29)
    x_label = np.split(x_label,29)
    x_label_train = np.split(x_label_train,29)
    y_length = np.split(y_length,29)
    y_length_train = np.split(y_length_train,29)
    return x_data, x_label, x_label_train, y_length, y_length_train

#tensorflow   

unit = 512
inputs = tf.placeholder(tf.float32,[None,80,4096]) 
labels = tf.placeholder(tf.int32,[None,max_seq_length])
labels_train = tf.placeholder(tf.int32,[None,max_seq_length])
length = tf.placeholder(tf.int32,[None])
length_train = tf.placeholder(tf.int32,[None])
batch_size = tf.shape(inputs)[0]
sequence_length = tf.fill([batch_size], max_seq_length)

def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(unit)

encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs, dtype=tf.float32)
print(encoder_outputs.get_shape())

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=unit, memory=encoder_outputs)

decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])
attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size)
initial_state = initial_state.clone(cell_state=encoder_state) 

embedding = tf.Variable(tf.random_uniform([len(encodeWords), unit], -0.1, 0.1, dtype=tf.float32))
labels_embedded = tf.nn.embedding_lookup(embedding, labels_train)

output_projection_layer = Dense(len(encodeWords), use_bias=False)

#train
helper = tf.contrib.seq2seq.TrainingHelper(labels_embedded, sequence_length)
#helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(labels_embedded, length_train,  0.5)
decoder = tf.contrib.seq2seq.BasicDecoder(attention_cell, helper, initial_state, output_layer=output_projection_layer)

decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_seq_length)

outputs = decoder_outputs.rnn_output
sample = decoder_outputs.sample_id

masks = tf.cast(tf.sequence_mask(length, maxlen=max_seq_length),tf.float32);
loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=labels, weights=masks,average_across_timesteps=False,average_across_batch=True)
loss = tf.reduce_sum(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
minimize = optimizer.minimize(loss)

trainCount = 0
totalLoss = 0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for j in range(800):
    x_data, x_label, x_label_train, y_length, y_length_train = getMiniDataSets()
    for i in range(29):
        trainCount = trainCount + 1
        
        _,l,predict = sess.run([minimize, loss, sample], 
                               feed_dict={
                                   inputs: x_data[i], 
                                   labels: x_label[i], 
                                   labels_train: x_label_train[i], 
                                   length: y_length[i],
                                   length_train: y_length_train[i]
                               })
        
        totalLoss += l
    ran = randint(0,49)
    log = "%d %f %s"%(j, totalLoss/trainCount, getStr(predict[ran]))
    print(log)
    if j%100==0:
        saver = tf.train.Saver()
        saver.save(sess, "model"+str(j)+".ckpt")
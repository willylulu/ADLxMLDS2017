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
            
wordsFreq = {}
for x in words:
    if x not in wordsFreq:
        wordsFreq[x] = 1
    else:
        wordsFreq[x] = wordsFreq[x] + 1
            
encodeWords = {}
counter = 3
freq = 1
        
for key,value in wordsFreq.items():
    if value>freq:
        encodeWords[key] = counter
        counter = counter + 1
encodeWords["<PAD>"] = 0      
encodeWords["<BOS>"] = 1
encodeWords["<EOS>"] = 2
encodeWords["something"] = 3
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
    sentence = sentence.replace('<BOS> ','').replace(' <EOS>', '').replace('<PAD>', '')
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
        
        temp = []
        for x in str.split(y," "):
            w = x.lower()
            if wordsFreq[w]>freq:
                temp.append(encodeWords[w])
            else:
                temp.append(encodeWords["something"])
                
        
        if(len(temp)>(max_seq_length-1)):
            temp = temp[:(max_seq_length-1)]
        
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

unit = 128
optimizer = tf.train.AdamOptimizer
inputs = tf.placeholder(tf.float32,[None,80,4096]) 
labels = tf.placeholder(tf.int32,[None,max_seq_length])
labels_train = tf.placeholder(tf.int32,[None,max_seq_length])
length = tf.placeholder(tf.int32,[None])
batch_size = tf.shape(inputs)[0]
inputs_length = tf.fill([batch_size],80)

w1 = tf.Variable( tf.random_uniform([4096, unit], -0.1, 0.1), name='w1')
b1 = tf.Variable( tf.zeros([unit]), name='b1')

w2 = tf.Variable( tf.random_uniform([unit, len(encodeWords)], -0.1, 0.1), name='w2')
b2 = tf.Variable( tf.zeros([len(encodeWords)]), name='b2')

l1 = tf.nn.rnn_cell.BasicLSTMCell(unit, state_is_tuple=True)
l2 = tf.nn.rnn_cell.BasicLSTMCell(unit, state_is_tuple=True)

print(l1.output_size)
print(l2.state_size.c)

att_w1 = tf.Variable( tf.random_uniform([l1.output_size, l1.output_size], -0.1, 0.1), name='aew1')
att_w2 = tf.Variable( tf.random_uniform([l2.state_size.h, l1.output_size], -0.1, 0.1), name='adw1')
#att_v = tf.Variable(tf.random_uniform([1, l1.output_size], -0.1, 0.1), name='attv')
w3 = tf.Variable( tf.random_uniform([l1.output_size, 1], -0.1, 0.1), name='w1')
b3 = tf.Variable( tf.zeros([1]), name='b1')

state1 = l1.zero_state(batch_size=batch_size, dtype=tf.float32)
state2 = l2.zero_state(batch_size=batch_size, dtype=tf.float32)


padding = tf.zeros([batch_size, unit],tf.float32)

embedding = tf.Variable(tf.random_uniform([len(encodeWords), unit], -0.1, 0.1), tf.float32, name='emb')

outputs = None
wordNums = None
l2_outputs = None

for i in range(80):
    
    d_output = tf.nn.xw_plus_b( inputs[:,i,:], w1, b1 )
    
    with tf.variable_scope("LSTM1") as scope:
        if i > 0 : scope.reuse_variables()
        output1, state1 = l1(tf.concat([padding, d_output],1), state1)
        
    with tf.variable_scope("LSTM2") as scope:
        output2, state2 = l2(tf.concat([padding, output1],1), state2)
        
        #attention
        l2_output = tf.expand_dims(output2, 1)
        l2_state = tf.expand_dims(state2, 1)
        l2_outputs = l2_output if l2_outputs is None else tf.concat([l2_outputs, l2_output], 1)
        
for i in range(0, max_seq_length):   
    
    att_ws = None
    
    for j in range(80):
        temp1 = tf.matmul(l2_outputs[:,j,:], att_w1) + tf.matmul(state2.h, att_w2)
        temp2 = tf.nn.xw_plus_b( tf.tanh(temp1), w3, b3 )
        temp3 = tf.nn.softmax(temp2, dim=-1)
        att_w = tf.expand_dims(temp3, 1)
        att_ws = att_w if att_ws is None else tf.concat([att_ws, att_w], 1)
        
    att_input = l2_outputs * att_ws
    att_input = tf.reduce_sum(att_input, 1)
    
    with tf.variable_scope("LSTM1"):
        output1, state1 = l1(tf.concat([att_input, padding],1), state1)
        
    embedded_input = tf.nn.embedding_lookup(embedding, labels_train[:, i]) 
        
    with tf.variable_scope("LSTM2"):
        output2, state2 = l2(tf.concat([embedded_input, output1],1), state2)   
        
    output = tf.nn.xw_plus_b( output2, w2, b2 )
    
    
    wordNum = tf.argmax(output, -1)
    output = tf.expand_dims(output, 1)
    wordNum = tf.expand_dims(wordNum, 1)
    
    outputs = output if outputs is None else tf.concat([outputs, output], 1)
    wordNums = wordNum if wordNums is None else tf.concat([wordNums, wordNum], 1)
    
    
masks = tf.cast(tf.sequence_mask(length, maxlen=max_seq_length),tf.float32);
loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=labels, weights=masks,average_across_timesteps=False,average_across_batch=True)
loss = tf.reduce_sum(loss)
minimize = optimizer(learning_rate=0.001).minimize(loss)



trainCount = 0
totalLoss = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
for j in range(150):
    x_data, x_label, x_label_train, y_length, y_length_train = getMiniDataSets()
    for i in range(29):
        trainCount = trainCount + 1
        
        _,l,predict = sess.run([minimize, loss, wordNums], 
                               feed_dict={
                                   inputs: x_data[i], 
                                   labels: x_label[i], 
                                   labels_train: x_label_train[i], 
                                   length: y_length[i]
                               })
        
        totalLoss += l
    ran = randint(0,49)
    log = "%d %f %s"%(j, totalLoss/trainCount, getStr(predict[ran]))
    print(log)
    if j%10==0 and j!=0:
        saver = tf.train.Saver()
        saver.save(sess, "model"+str(j)+".ckpt")
        
saver = tf.train.Saver()
saver.save(sess, "model"+str(j)+".ckpt")
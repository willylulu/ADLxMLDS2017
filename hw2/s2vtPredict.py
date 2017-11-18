import os
import numpy as np
import json
import string
from random import randint
import sys

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

path = sys.argv[1]

testdir = "testing_data/feat/"

testfiles = os.listdir(path+testdir)

testdata = {}
videos = []
for i in range(0,len(testfiles)):
    name = str.split(testfiles[i],".")[0]+'.'+str.split(testfiles[i],".")[1]
    testdata[name] = np.load(path+testdir+testfiles[i])
    videos.append(name)
print(len(testdata))

testjsonfile = open(path+"testing_label.json","r")
testjson = json.load(testjsonfile)

# ddd = json.load(open("decoder.json"))
# decodeWords = {}
# encodeWords = {}
# for key, value in ddd.items():
#     decodeWords[int(key)] = value
#     encodeWords[value] = key

decodeWords = np.load("decodeWords.npy").item()
encodeWords = np.load("encodeWords.npy").item()

max_seq_length = 21

def getStr(ints):
    temp = []
    temp2 = None
    for x in ints:
        if x != temp2:
            temp.append(x)
            temp2 = x
            #hold[x] = True
        if x == encodeWords["<EOS>"]:
            temp.append(x)
            break
    sentence = ' '.join([decodeWords[int(x)] for x in temp])
    sentence = sentence.replace('<EOS> ','').replace(' <EOS>', '')
    return sentence

def getTestDataSets():
    x_data = np.zeros((100,80,4096),dtype="float32")

    i = 0
    for x in testjson:
        name = x["id"]
        temp = testdata[name]
        x_data[i] = temp
        i=i+1
    return x_data

unit = 256
inputs = tf.placeholder(tf.float32,[None,80,4096])
batch_size = tf.shape(inputs)[0]
start_tokens = tf.fill([batch_size], encodeWords["<BOS>"])
inputs_length = tf.fill([batch_size],80)
sequence_length = tf.fill([batch_size], max_seq_length)

w1 = tf.Variable( tf.random_uniform([4096, unit], -0.1, 0.1), name='w1')
b1 = tf.Variable( tf.zeros([unit]), name='b1')

w2 = tf.Variable( tf.random_uniform([unit, len(encodeWords)], -0.1, 0.1), name='w2')
b2 = tf.Variable( tf.zeros([len(encodeWords)]), name='b2')

l1 = tf.nn.rnn_cell.GRUCell(unit)
#l1 = tf.nn.rnn_cell.DropoutWrapper(l1, output_keep_prob=1,variational_recurrent=True, dtype=tf.float32)
l2 = tf.nn.rnn_cell.GRUCell(unit)
#l2 = tf.nn.rnn_cell.DropoutWrapper(l2, output_keep_prob=1,variational_recurrent=True, dtype=tf.float32)

att_w1 = tf.Variable( tf.random_uniform([l1.output_size, l1.output_size], -0.1, 0.1), name='aew1')
att_w2 = tf.Variable( tf.random_uniform([l2.state_size, l1.output_size], -0.1, 0.1), name='adw1')
att_v = tf.Variable(tf.random_uniform([l1.output_size, 1], -0.1, 0.1), name='attv')
# w3 = tf.Variable( tf.random_uniform([l1.output_size, 1], -0.1, 0.1), name='w1')
# b3 = tf.Variable( tf.zeros([1]), name='b1')

state1 = l1.zero_state(batch_size=batch_size, dtype=tf.float32)
state2 = l2.zero_state(batch_size=batch_size, dtype=tf.float32)


padding = tf.zeros([batch_size, unit],tf.float32)

embedding = tf.Variable(tf.random_uniform([len(encodeWords), unit], -0.1, 0.1), tf.float32, name='emb')

# w2v = np.load("w2v3.npy")
# init = tf.constant_initializer(w2v)
#embedding = tf.get_variable("embedding", shape=[len(encodeWords), unit], initializer=init, dtype=tf.float32, trainable=False)
#embedding = tf.convert_to_tensor(w2v, np.float32)

outputs = None
wordNums = None
encode_outputs = None
l2_outputs = None

intput_temp1 = tf.reshape(inputs, [-1, 4096])
intput_temp2 = tf.nn.xw_plus_b( intput_temp1, w1, b1 )
d_output = tf.reshape(intput_temp2, [ batch_size, 80, unit])

for i in range(80):
    
    #d_output = tf.nn.xw_plus_b( inputs[:,i,:], w1, b1 )
    
    with tf.variable_scope("LSTM1") as scope:
        if i > 0 : scope.reuse_variables()
        output1, state1 = l1(tf.concat([padding, d_output[:,i,:]],1), state1)
        
    with tf.variable_scope("LSTM2") as scope:
        output2, state2 = l2(tf.concat([padding, output1],1), state2)
        
        #attention
        att_tar = output2
        l2_output = tf.expand_dims(att_tar, 1)
        l2_state = tf.expand_dims(state2, 1)
        l2_outputs = l2_output if l2_outputs is None else tf.concat([l2_outputs, l2_output], 1)
        
for i in range(0, max_seq_length):   
    
    if i == 0:
        embedded_input = tf.nn.embedding_lookup(embedding, tf.ones([batch_size], tf.int32))
    
    att_ws = None

    att_temp1 = tf.reshape(l2_outputs, [-1, unit])
    att_temp2 = tf.matmul(att_temp1, att_w1)
    att_temp3 = tf.matmul(state2, att_w2)
    att_temp4 = tf.tile(att_temp3, [80,1])
    att_temp5 = tf.matmul(tf.tanh(att_temp2 + att_temp4), att_v)
    att_temp6 = tf.nn.softmax(att_temp5)
    att_temp7 = tf.tile(att_temp6, [1,unit])
    att_ws = tf.reshape(att_temp7, [batch_size, 80, unit])
    att_input = l2_outputs * att_ws
    att_input = tf.reduce_sum(att_input, 1)
    
    with tf.variable_scope("LSTM1"):
        output1, state1 = l1(tf.concat([att_input, padding],1), state1)
        
    with tf.variable_scope("LSTM2"):
        output2, state2 = l2(tf.concat([embedded_input, output1],1), state2)  
    
    output = tf.nn.xw_plus_b( output2, w2, b2 )
    
    wordNum = tf.argmax(output, -1)
    embedded_input = tf.nn.embedding_lookup(embedding, wordNum)                                             
                                                
    output = tf.expand_dims(output, 1)
    wordNum = tf.expand_dims(wordNum, 1)
    
    outputs = output if outputs is None else tf.concat([outputs, output], 1)
    wordNums = wordNum if wordNums is None else tf.concat([wordNums, wordNum], 1)  
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess, "model2/model57.ckpt")

x_data = getTestDataSets()
predict = sess.run([wordNums], feed_dict={inputs: x_data})
ans = [getStr(x) for x in predict[0]]
print(sys.argv[2])
f = open(sys.argv[2], "w")
for i in range(0,len(ans)):
    f.write(videos[i]+","+ans[i]+"\n")
f.close()
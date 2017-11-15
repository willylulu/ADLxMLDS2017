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

encodeWords = np.load("encodeWords.npy").item()
decodeWords = np.load("decodeWords.npy").item()
decodeWords[-1] = ""
decodeWords[encodeWords["something"]] = ""
print(len(encodeWords))
print(len(decodeWords))

max_seq_length = 21

def getStr(ints):
    sentence = ' '.join([decodeWords[int(x)] for x in ints])
    sentence = sentence.replace('<BOS> ','').replace(' <EOS>', '')
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

unit = 512
inputs = tf.placeholder(tf.float32,[None,80,4096])
batch_size = tf.shape(inputs)[0]
start_tokens = tf.fill([batch_size], encodeWords["<BOS>"])
inputs_length = tf.fill([batch_size],80)
sequence_length = tf.fill([batch_size], max_seq_length)

w1 = tf.Variable( tf.random_uniform([4096, unit], -0.1, 0.1), name='w1')
b1 = tf.Variable( tf.zeros([unit]), name='b1')

w2 = tf.Variable( tf.random_uniform([unit, len(encodeWords)], -0.1, 0.1), name='w2')
b2 = tf.Variable( tf.zeros([len(encodeWords)]), name='b2')

l1 = tf.nn.rnn_cell.BasicLSTMCell(unit)
l2 = tf.nn.rnn_cell.BasicLSTMCell(unit)

state1 = l1.zero_state(batch_size=batch_size, dtype=tf.float32)
state2 = l2.zero_state(batch_size=batch_size, dtype=tf.float32)


padding = tf.zeros([batch_size, unit],tf.float32)
padding2 = tf.zeros([batch_size, max_seq_length, unit],tf.float32)

embedding = tf.Variable(tf.random_uniform([len(encodeWords), unit], -0.1, 0.1), tf.float32, name='emb')

outputs = None
wordNums = None
encode_outputs = None

for i in range(0, 80):
    
    d_output = tf.nn.xw_plus_b( inputs[:,i,:], w1, b1 )
    
    with tf.variable_scope("LSTM1") as scope:
        if i > 0 : scope.reuse_variables()
        output1, state1 = l1(d_output, state1)
        
    with tf.variable_scope("LSTM2") as scope:
        output2, state2 = l2(tf.concat([padding, output1],1), state2)
        
for i in range(0, max_seq_length):
    if i == 0:
        embedded_input = tf.nn.embedding_lookup(embedding, tf.ones([batch_size], tf.int32))    
    
    with tf.variable_scope("LSTM1"):
        output1, state1 = l1(padding, state1)
        
    with tf.variable_scope("LSTM2"):
        output2, state2 = l2(tf.concat([embedded_input, output1],1), state2)   
        
    output = tf.nn.xw_plus_b( output2, w2, b2 )
    
    wordNum = tf.argmax(output, -1)
    embedded_input = tf.nn.embedding_lookup(embedding, wordNum)                                             
                                                
    output = tf.expand_dims(output, 1)
    wordNum = tf.expand_dims(wordNum, 1)
    
    outputs = output if outputs is None else tf.concat([outputs, output], 1)
    wordNums = wordNum if wordNums is None else tf.concat([wordNums, wordNum], 1)  

for iii in range(1,71):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(sess, "model"+str(iii*10)+".ckpt")

    x_data = getTestDataSets()
    predict = sess.run([wordNums], feed_dict={inputs: x_data})
    ans = [getStr(x) for x in predict[0]]
    print("answer"+str(iii*10)+".txt")
    f = open("answer"+str(iii*10)+".txt","w")
    for i in range(0,len(ans)):
        f.write(videos[i]+","+ans[i]+"\n")
    f.close()
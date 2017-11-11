import os
import numpy as np
import json
import string
import sys
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

encodeWords = np.load("encodeWords.npy").item()
decodeWords = np.load("decodeWords.npy").item()
decodeWords[-1] = ""
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
beam_width = 3
start_tokens = tf.fill([batch_size], encodeWords["<BOS>"])

def lstm_cell():
  return tf.contrib.rnn.BasicLSTMCell(unit)

encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])

encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inputs, dtype=tf.float32)

tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, beam_width)

tiled_encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)

tiled_sequence_length = tf.contrib.seq2seq.tile_batch(tf.fill([batch_size],80), beam_width)

batch_size_beam = tf.shape(tiled_encoder_outputs)[0]

attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=unit, memory=tiled_encoder_outputs, memory_sequence_length = tiled_sequence_length)

decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)])

attention_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism)

initial_state = attention_cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
initial_state = initial_state.clone(cell_state=tiled_encoder_state) 

embedding = tf.Variable(tf.random_uniform([len(encodeWords), unit], -0.1, 0.1, dtype=tf.float32))

output_projection_layer = Dense(len(encodeWords), use_bias=False)

#train
#helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding, start_tokens, encodeWords["<EOS>"])
decoder = tf.contrib.seq2seq.BeamSearchDecoder(attention_cell, embedding, start_tokens, encodeWords["<EOS>"], initial_state, beam_width, output_layer=output_projection_layer,
        length_penalty_weight=0.0)
decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_seq_length, impute_finished=False)

outputs = decoder_outputs.predicted_ids[:,:,0]

print("I think it's OK!")

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

saver.restore(sess, "model200.ckpt")

x_data = getTestDataSets()
predict = sess.run([outputs], feed_dict={inputs: x_data})

ans = [getStr(x) for x in predict[0]]

f = open(sys.argv[2],"w")
for i in range(0,len(ans)):
    name = videos[i]
    if name == 'klteYv1Uv9A_27_33.avi' or name == '5YJaS2Eswg0_22_26.avi' or name == 'UbmZAe5u5FI_132_141.avi' or name=='JntMAcTlOF0_50_70.avi' or name=='tJHUH9tpqPg_113_118.avi':
        f.write(videos[i]+","+ans[i]+"\n")
f.close()
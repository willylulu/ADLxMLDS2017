import os
import numpy as np
import json
import string
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Flatten, Reshape, TimeDistributed
from keras.layers import recurrent
import keras.optimizers

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

maxcounter = 2
maxlen = -1
seqnum = 0
words = []
for x in trainjson:
    counter2 = 0
    for y in x['caption']:
        seqnum = seqnum + 1
        if counter2 == maxcounter:
            break
        counter2 = counter2 + 1
        y = ''.join(c for c in y if c not in string.punctuation)
        ss = unicode.split(y," ")
        if len(ss)>maxlen:
            maxlen = len(ss)
            maxlenStr = y
        for z in ss:
            words.append(z.lower())
print(maxlen)
print(maxlenStr)
print(seqnum)

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

x_data = np.zeros((seqnum,80,4096))
x_label = np.zeros((seqnum,maxlen,1))
i = 0;
for x in trainjson:
    name = x["id"]
    temp = traindata[name]
    counter2 = 0
    for y in x["caption"]:
        if counter2 == maxcounter:
            break
        counter2 = counter2 + 1
        x_data[i] = temp
        i = i+1
        tempB = []
        y = ''.join(c for c in y if c not in string.punctuation)
        for z in unicode.split(y," "):
            tempB.append(encodeWords[z.lower()])
        for xa in range(len(tempB),maxlen):
            tempB.append(encodeWords["<EOS>"])
        tempB = np.reshape(tempB,(maxlen,1))
        x_label[i] = tempB
print(x_data.shape)
print(x_label.shape)
np.save("x_data.npy",x_data)
np.save("x_label.npy",x_label)
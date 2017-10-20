import os
import numpy as np
import math

print("load train.ark")
mfcc_train_rawData = open("../../data/mfcc/train.ark","r").read().splitlines()
#mfcc_test_rawData = open("../../data/mfcc/test.ark","r").readlines()
print("load train.lab")
train_label_rawData = open("../../data/train.lab","r").read().splitlines()
print(len(mfcc_train_rawData))
# print(len(mfcc_test_rawData))
print(len(train_label_rawData))

print("load 48_39.map")
map48_39_raw = open("../../data/48_39.map").read().splitlines()
map48_39 = {}
for x in map48_39_raw:
    tempX = x.split('\t')
    map48_39[tempX[0]] = tempX[1]

print("Initial mfcc_train_data matrix")
mfcc_train_data = np.zeros((len(mfcc_train_rawData),39))
mfcc_train_data_index = {}
train_label = np.zeros(len(mfcc_train_rawData))

# mfcc_test_data = []
# mfcc_test_data_index = {}

print("update mfcc_train_data matrix")
for i in range(0,len(mfcc_train_rawData)):
    tempStr = mfcc_train_rawData[i]
    tempX = tempStr.split( )
    for j in range(0,len(mfcc_train_data[i])):
        mfcc_train_data[i][j] = float(tempX[j+1])
    mfcc_train_data_index[tempX[0]] = i

print(len(mfcc_train_data[0]))

map39 = {}
count = 0

print("transform phone to number")
for i in range(0,len(train_label_rawData)):
    tempStr = train_label_rawData[i]
    tempX = tempStr.split(',')
    label = map48_39[tempX[1]]
    if label not in map39:
        map39[label] = count
        count = count + 1
    train_label[mfcc_train_data_index[tempX[0]]] = map39[label]

print(map39)

print("resize")
timeStep = 123
# int(math.ceil(mfcc_train_data.shape[0]/timeStep))
# int(math.ceil(train_label.shape[0]/timeStep))
mfcc_train_data = np.resize(mfcc_train_data,(int(math.ceil(mfcc_train_data.shape[0]/timeStep)),timeStep,39))
train_label = np.resize(train_label,(int(math.ceil(train_label.shape[0]/timeStep)),timeStep))
print(mfcc_train_data.shape)
print(train_label.shape)
print(train_label[0])

print("save files")
np.save("map39",map39)
mfcc_train_data.dump("../../data/mfcc_x.dat")
train_label.dump("../../data/y.dat")
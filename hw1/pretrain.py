import os
import numpy as np

mfcc_train_rawData = open("../../data/mfcc/train.ark","r").readlines()
#mfcc_test_rawData = open("../../data/mfcc/test.ark","r").readlines()
train_label_rawData = open("../../data/train.lab","r").readlines()
print(len(mfcc_train_rawData))
# print(len(mfcc_test_rawData))
print(len(train_label_rawData))

mfcc_train_data = np.zeros((len(mfcc_train_rawData),39))
mfcc_train_data_index = {}
train_label = np.chararray(len(mfcc_train_rawData))

# mfcc_test_data = []
# mfcc_test_data_index = {}

for i in range(0,len(mfcc_train_rawData)):
    tempStr = mfcc_train_rawData[i]
    tempX = tempStr.split( )
    for j in range(0,len(mfcc_train_data[i])):
        mfcc_train_data[i][j] = float(tempX[j+1])
    mfcc_train_data_index[tempX[0]] = i

print(len(mfcc_train_data[0]))

# for i in range(0,len(mfcc_test_rawData)):
#     tempStr = mfcc_test_rawData[i]
#     tempX = tempStr.split( )
#     mfcc_test_data.append(tempX[1:])
#     mfcc_test_data_index[tempX[0]] = i

# print(len(mfcc_test_data[0]))

for i in range(0,len(train_label_rawData)):
    tempStr = train_label_rawData[i]
    tempX = tempStr.split(',')
    train_label[mfcc_train_data_index[tempX[0]]] = tempX[1]

mfcc_train_data.dump("../../data/mfcc_x.dat")
train_label.dump("../../data/y.dat")
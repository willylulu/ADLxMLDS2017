import os
import sys
import numpy as np
import keras
from keras.models import load_model
import math

print("load train.lab")
train_label_rawData = open(sys.argv[1]+"train.lab","r").read().splitlines()
print(len(train_label_rawData))

print("load 48_39.map")
map48_39_raw = open(sys.argv[1]+"48_39.map").read().splitlines()
map48_39 = {}
for x in map48_39_raw:
    tempX = x.split('\t')
    map48_39[tempX[0]] = tempX[1]

# mfcc_test_data = []
# mfcc_test_data_index = {}

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
np.save("map39",map39)

model = load_model('best_model.h5')
model.summary()

#print("load test.ark")
mfcc_test_rawData = open(sys.argv[1]+"mfcc/test.ark","r").read().splitlines()
mfcc_test_data = np.zeros((len(mfcc_test_rawData),39))
mfcc_test_data_index = {}

#print("Make mfcc_test_data matrix")
for i in range(0,len(mfcc_test_rawData)):
    tempStr = mfcc_test_rawData[i]
    tempX = tempStr.split( )
    for j in range(0,len(mfcc_test_data[i])):
        mfcc_test_data[i][j] = float(tempX[j+1])
    mfcc_test_data_index[tempX[0]] = i


# map48_39_raw = open("../../data/48_39.map").read().splitlines()
# map48_39 = {}
# for x in map48_39_raw:
#     tempX = x.split('\t')
#     map48_39[tempX[0]] = tempX[1]

#print("load 48phone_char.map")
phone_char48 = open(sys.argv[1]+"48phone_char.map").read().splitlines()
phoneToChar = {}
for x in phone_char48:
    tempX = x.split('\t')
    phoneToChar[tempX[0]] = tempX[2]

#print("load map39.npy")
map39 = np.load("map39.npy").item()
map39Inverse = {}
for key, value in map39.items():
    map39Inverse[value] = key

#print("predict")

timeStep = 123
mfcc_test_data = np.resize(mfcc_test_data,(int(math.ceil(mfcc_test_data.shape[0]/timeStep)),timeStep,39))

resultA = model.predict(mfcc_test_data, batch_size=128)
resultA = np.resize(resultA,(len(mfcc_test_rawData),39))
resultA = np.max(resultA, axis=1)
#print(resultA[0:100])

result = model.predict_classes(mfcc_test_data, batch_size=128, verbose=0)
result = np.resize(result,(len(mfcc_test_rawData)))

#print(resultA.shape)
#print(result.shape)

mfcc_test_string = {}

#print("trim")
for key, value in mfcc_test_data_index.items():
    temp = key.split("_")
    instance = temp[0] + "_" + temp[1]
    number = int(temp[2])
    if instance not in mfcc_test_string:
        mfcc_test_string[instance] = {}
    ans = result[mfcc_test_data_index[key]]
    ans = map39Inverse[ans]
    if resultA[value] < 0.6:
        ans = ""
    mfcc_test_string[instance][number-1] = ans

for key, value in mfcc_test_string.items():
    trim = [None]*len(value)

    for key2, value2 in value.items():
        trim[key2] = value2
    
    temp = None
    trim2 = []

    for x in trim:
        if x == temp:
            continue
        else:
            trim2.append(x)
            temp = x

    sil = False
    trim3 = []
    for x in trim2:
        if sil == True:
            trim3.append(x)
        else:
            if x != "sil":
                trim3.append(x)
                sil = True

    for i in range(len(trim3)-1,-1,-1):
        if trim3[i] != "sil":
            trim3 = trim3[0:i+1]
            break
    
    trim4 = []
    for i in range(0,len(trim3)):
        if trim3[i] != "":
            trim4.append(phoneToChar[trim3[i]])

    mfcc_test_string[key] = trim4

#print("trim again")
for key, value in mfcc_test_string.items():
    trim = value
    
    temp = None
    trim2 = []

    for x in trim:
        if x == temp:
            continue
        else:
            trim2.append(x)
            temp = x

    sil = False
    trim3 = []
    for x in trim2:
        if sil == True:
            trim3.append(x)
        else:
            if x != "sil":
                trim3.append(x)
                sil = True

    for i in range(len(trim3)-1,-1,-1):
        if trim3[i] != "sil":
            trim3 = trim3[0:i+1]
            break

    mfcc_test_string[key] = trim3

#print("make answer and save file")
ansFile = open(sys.argv[2]+"best.csv","w")
ansFile.write("id,phone_sequence\n")

for key, value in mfcc_test_string.items():
    temp = ""
    for x in value:
        temp += x
    ansFile.write(key+","+temp+"\n")
ansFile.close()

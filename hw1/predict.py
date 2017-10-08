import os
import numpy as np
import keras
from keras.models import load_model

model = load_model('mfcc_model.h5')

mfcc_test_rawData = open("../../data/mfcc/test.ark","r").read().splitlines()
mfcc_test_data = np.zeros((len(mfcc_test_rawData),39))
mfcc_test_data_index = {}

for i in range(0,len(mfcc_test_rawData)):
    tempStr = mfcc_test_rawData[i]
    tempX = tempStr.split( )
    for j in range(0,len(mfcc_test_data[i])):
        mfcc_test_data[i][j] = float(tempX[j+1])
    mfcc_test_data_index[tempX[0]] = i

phone_char48 = open("../../data/48phone_char.map").read().splitlines()
numToChar = {}
numToPhone = {}
phoneToChar = {}
for x in phone_char48:
    tempX = x.split('\t')
    numToPhone[int(tempX[1])] = tempX[0]
    numToChar[int(tempX[1])] = tempX[2]
    phoneToChar[tempX[0]] = tempX[2]

map48_39_raw = open("../../data/48_39.map").read().splitlines()
map48_39 = {}
for x in map48_39_raw:
    tempX = x.split('\t')
    map48_39[tempX[0]] = tempX[1]

result = model.predict_classes(mfcc_test_data, batch_size=8192, verbose=0)

mfcc_test_string = {}

for key, value in mfcc_test_data_index.items():
    temp = key.split("_")
    instance = temp[0] + "_" + temp[1]
    number = int(temp[2])
    if instance not in mfcc_test_string:
        mfcc_test_string[instance] = {}
    ans = result[mfcc_test_data_index[key]]
    ans = numToPhone[ans]
    ans = map48_39[ans]
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
    
    for i in range(0,len(trim3)):
        trim3[i] = phoneToChar[trim3[i]]

    mfcc_test_string[key] = trim3

ansFile = open("ans.csv","w")
ansFile.write("id,phone_sequence\n")

for key, value in mfcc_test_string.items():
    temp = ""
    for x in value:
        temp += x
    ansFile.write(key+","+temp+"\n")
ansFile.close()

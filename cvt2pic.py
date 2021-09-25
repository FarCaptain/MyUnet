# coding: utf-8


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras

import cv2


print(tf.__version__)
print(sys.version_info)


#convert all my data to pictures
def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss

#224*1016 -> 224*800 scaled pictures
def read_file(index):
    mat = np.array([[0]*800 for i in range(224) ])
    string = "D:\FarCaptain\practice\SaveData\data_"+str(index)+".txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        line = line[:800]
        for x in line:
            mat[i][j] = x +128 #标准化0~255
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

def read_label(index):
    mat =np.array([[0]*800 for i in range(224) ])
    string = "D:\FarCaptain\practice\SaveData\data_"+str(index)+"_label.txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        line = line[:800]
        for x in line:
            mat[i][j] = x*255
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

savedir = "D:\FarCaptain\practice\saveLabel"
for i in range(236,721):
    string = "D:\FarCaptain\practice\SaveData\data_" + str(i) + "_label.txt"
    if not os.path.exists(string):
        continue
    #mat = read_file(i) #get image array
    mat = read_label(i) #get label array
    img = np.array(mat, dtype=np.uint8)
    print(img.max())
    #print(img.shape)
    #cv2.imshow("wtf",img)
    #cv2.waitKey(0)
    #print(img)
    #plt.imshow(mat, cmap='binary')
    #plt.show()
    savepath = savedir+"\label_"+str(i)+".png"
    cv2.imwrite(savepath,img)


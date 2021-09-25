import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
from sklearn.model_selection import train_test_split

def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss

def read_file(index):
    mat = np.array([[0]*1016 for i in range(224) ])
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+".txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        for x in line:
            mat[i][j] = x
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

def read_label(index):
    mat =np.array([[0]*1016 for i in range(224) ])
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+"_label.txt"
    f = open(string,"r")
    line = f.readline()
    i = 0
    while line:
        j = 0
        line = get_number_list(line)
        for x in line:
            mat[i][j] = x
            j+=1
        i+=1
        line = f.readline()
    f.close()
    return mat

images = []
labels = []
for i in range(1,721):
    images.append(read_file(i))
    string = "D:\\0Sphinx\\SaveData\\data_"+str(i)+"_label.txt"
    if os.path.exists(string):
        labels.append(read_label(i))
    else:
        labels.append(read_label(78)) #零矩阵

X = np.array(images) #数据集
print(X.shape)

Y = np.array(labels)
print(Y.shape)

#split into train,valid,test
x_train_all, x_test, y_train_all, y_test = train_test_split(X,Y,test_size=0.2,random_state=30)

x_valid, x_train = x_train_all[:432], x_train_all[432:]


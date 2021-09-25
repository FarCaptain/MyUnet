import os
import numpy as np

def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss

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

def left_point(i,j,mat): #这个点是左上角
    if(i==0 or mat[i-1][j]==0):
        p = i
        q = j
        while mat[p][q]==1:
            q = j
            while mat[p][q]==1:
                q+=1
            p+=1
        #print((p-i)*(q-j))
        return (p-i)*(q-j)
    else:
        return -1

max_sz = -10
min_sz = 200000
for index in range(1,721):
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+"_label.txt"
    if os.path.exists(string):
        mat =np.array([[0]*1016 for i in range(224) ])
        mat = read_label(index)

        flag = False
        for i in range(224):
            for j in range(1016):
                val = mat[i][j]
                if(val==1):
                    if flag==False:
                        flag = True
                        tmp = left_point(i,j,mat)
                        if(tmp!=-1):
                            max_sz = max(max_sz,left_point(i,j,mat))
                            min_sz = min(min_sz,left_point(i,j,mat))
                elif(flag==True):flag = False





print(min_sz,"~",max_sz)

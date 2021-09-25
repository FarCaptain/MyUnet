import matplotlib.pyplot as plt

def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss


mat = [[0]*1016 for i in range(224) ]
num = 240
string = "D:\\0Sphinx\\SaveData\\data_"+str(num)+"_label.txt"
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
plt.imshow(mat,cmap='binary')
plt.show()

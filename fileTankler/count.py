import os
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss


cnt = 0 #有瑕疵的数据个数
for num in range(1,721):
    string = "D:\\0Sphinx\\SaveData\\data_"+str(num)+"_label.txt"

    flag = True #完好无损
    if os.path.exists(string):
        #print(string)
        f = open(string, "r")
        line = f.readline()
        line = line[:-1] #除掉尾巴
        while line and flag:
            for nm in line:
                if nm == '1': flag = False #有瑕疵
            if not flag:
                cnt += 1
                break
            line = f.readline()
            line = line[:-1]
        f.close()

    print(num,": ",flag)
else: print(cnt)

'''
string = "D:\\0Sphinx\\SaveData\\data_78_label.txt"
f = open(string,"r")
line = f.readline()
#l = len(line)
l_cnt = 0
r_cnt = 0
for ch in line:
    if ch == '1' or ch=='0':
        r_cnt+=1
while line:
    l_cnt+=1
    line = f.readline()
f.close()
print(l_cnt," lines")
print(r_cnt," rows")


#计算图片pixel value range

mx = 0
mn = 0
for num in range(1,721):
    string = "D:\\0Sphinx\\SaveData\\data_"+str(num)+".txt"
    f = open(string,"r")
    line = f.readline()
    while line:
        line = get_number_list(line)
        for x in line:
            mx = max(mx,x)
            mn = min(mn,x)

        line = f.readline()
    f.close()
print(mn,"~",mx)
'''




# 总共 720 个
# 191 个有瑕疵的数据
# 529 个没毛病的数据
#-128~126

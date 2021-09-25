import os


def get_number_list(ss):
    ss = ss.split(',')
    ss = [int(x) for x in ss]
    return ss

min_l = 10000
max_l = -1
min_r = 10000
max_r = -1
for index in range(1,721):
    string = "D:\\0Sphinx\\SaveData\\data_"+str(index)+"_label.txt"
    if os.path.exists(string):
        f = open(string,"r")

        line = f.readline()
        i = 0
        while line:
            j = 0
            line = get_number_list(line)
            for x in line:
                if x==1:
                    min_l = min(min_l,i)
                    max_l = max(max_l,i)
                    min_r = min(min_r,j)
                    max_r = max(max_r,j)
                j+=1
            i+=1
            line = f.readline()
        f.close()

print(min_l,"~",max_l)
print(min_r,"~",max_r)
#38 ~ 78
#141 ~ 682

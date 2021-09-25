import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
import random
from keras import backend as K
from tensorflow.contrib.opt import AdamWOptimizer

import matplotlib as mpl
import matplotlib.pyplot as plt


import hashlib
from PyQt5.QtSql import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # CPU上tf优化

class detectDialog(QDialog):
    detect_success_signal = pyqtSignal()
    logdir = './savedModels'

    def __init__(self, parent=None):
        super(detectDialog, self).__init__(parent)
        self.setUpUI()
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("检测图像")
        self.uple = []
        self.dnri = []
        self.cnt = 0 #rec

    def setUpUI(self):
        self.resize(700, 800)
        #print(self.fname)

    def setInfo(self, str):
        self.pic_name = str[:-4]
        print(self.pic_name)


        self.gray = cv2.imread(str, cv2.IMREAD_GRAYSCALE)
        print(self.gray.shape)


    def detectImage(self):
        # 检测图像，得到输出

        #1 读入模型
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
        modir =""
        #check the setting file to determine current Model
        with open("./setting.txt", 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                modir = lines

        self.output_model_file = modir
        loaded_model = keras.models.load_model(self.output_model_file)

        inpt = np.array(self.gray).reshape(-1,224,800,1)
        res = loaded_model.predict(inpt)

        res = res.reshape(224,800)
        res*=255.0
        #res = self.gray
        img = np.array(res, dtype=np.uint8)
        print(img)
        print(img.shape)
        self.processMatrix(img)
        #plt.imshow(img, cmap='gray')
        #plt.show()


        #return rst
    def processMatrix(self, mat):
        vis = np.array([[False] * 800 for i in range(224)])
        for i in range(0,224):
            for j in range(0,800):
                if mat[i][j] > 50 and vis[i][j]==False:#有区域，确认区域大小
                    mp = i
                    mq = j
                    for p in range(i,224):
                        if mat[p][j] <= 50:
                            mp = max(mp,p-1)
                            break
                        for q in range(j,800):
                            if mat[p][q]<=50:
                                mq = max(mq,q-1)
                                break
                            else: vis[p][q] = True
                    self.uple.append((i,j))
                    self.dnri.append((mp,mq))
                    #for p in range
        # convert gray image to RGB
        self.grayBGR = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)

        # Draw the colored rectangle
        ll = len(self.uple)
        for i in range(0,ll):
            up = self.uple[i]
            dn = self.dnri[i]
            cv2.rectangle(self.grayBGR, up, dn, (0,255,255), 2)
        #cv2.imshow("tst", self.grayBGR)
        print(self.grayBGR)
        plt.imshow(self.grayBGR)
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = detectDialog()
    mainMindow.show()
    sys.exit(app.exec_())
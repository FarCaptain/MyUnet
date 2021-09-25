# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import qdarkstyle
from PyQt5.QtSql import *
from LabDialog import LabDialog


class ExpertHome(QWidget):

    def __init__(self):
        super(ExpertHome, self).__init__()
        self.resize(700, 500)
        self.setWindowTitle("欢迎，专家用户")
        self.setUpUI()

    def __del__(self):
        pass

    def setUpUI(self):
        # assign a existed Model
        layout = QVBoxLayout()

        modleName = self.getModel()
        modleName = modleName[:-3]
        txt = "当前使用模型：" + modleName
        self.textLable = QLabel(txt)
        layout.addWidget(self.textLable)

        self.modButton = QPushButton('更改指定模型')
        self.modButton.clicked.connect(self.loadModel)
        layout.addWidget(self.modButton)

        self.newButton = QPushButton('☯训练新模型')
        self.newButton.clicked.connect(self.trainModel)
        layout.addWidget(self.newButton)

        self.setLayout(layout)

    def loadModel(self):
        # 从文件导入模型
        self.fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '模型文件(*.h5)')
        if self.fname == '':
            return
        li = len(self.fname)
        st = 0
        for x in range(li-1,0,-1):
            if self.fname[x] == '/':
                st = x
                break
        print(self.fname[st+1:])
        modName = self.fname[st+1:-3]
        #显示新导入的文件名字
        self.textLable.setText('当前使用模型：'+modName)

        #更改配置文件
        f = open("./setting.txt", 'w')
        f.write(self.fname)
        f.close()
        return

    def getModel(self):
        #get model name
        modir = ""
        # check the setting file to determine current Model
        with open("./setting.txt", 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                modir = lines
        file_to_read.close()
        li = len(modir)
        print(modir)
        st = 0
        for x in range(li-1,0,-1):
            if modir[x] == '/':
                st = x
                break
        print(modir[st+1:])
        return modir[st+1:]

    def trainModel(self):
        Lab = LabDialog(self)
        Lab.show()
        Lab.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = ExpertHome()
    mainMindow.show()
    sys.exit(app.exec_())
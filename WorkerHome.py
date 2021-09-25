# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
from PyQt5.QtSql import *
from detectDialog import detectDialog


class WorkerHome(QWidget):


    def __init__(self):
        super(WorkerHome, self).__init__()
        self.resize(1000, 800)
        self.setWindowTitle("欢迎使用Weldetect焊件检测系统")
        self.setUpUI()
        self.f = None
        self.fname = ""
        self.fname_signal = pyqtSignal(str)  # 传走文件名


    def __del__(self):
        pass

    def setUpUI(self):
        layout = QVBoxLayout()
        self.picButton = QPushButton('导入图片')
        self.picButton.clicked.connect(self.loadImage)
        layout.addWidget(self.picButton)

        self.imageLable = QLabel()
        layout.addWidget(self.imageLable)

        self.submitButton = QPushButton('提交检测')
        self.submitButton.clicked.connect(self.submit)
        layout.addWidget(self.submitButton)

        self.setLayout(layout)

    def loadImage(self):
        self.fname,_ = QFileDialog.getOpenFileName(self,'打开文件','.','图像文件(*.jpg *.png)')
        pix = QPixmap(self.fname)
        self.imageLable.setPixmap(pix)
        if self.fname=='':
            return
        self.f = open(self.fname,'r')
        return

    def submit(self):
        print(self.f)
        if self.f == None or self.fname=='':
            print(QMessageBox.warning(self, "错误", "请先导入图像文件!", QMessageBox.Yes, QMessageBox.Yes))
            return
        #self.fname_signal.emit(self.fname)

        #将 self.f 放进模型检测...
        _detectDialog = detectDialog(self)
        _detectDialog.setInfo(self.fname)
        _detectDialog.detectImage()
        #_detectDialog.show()
        #_detectDialog.exec_()
        return



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = WorkerHome()
    mainMindow.show()
    sys.exit(app.exec_())
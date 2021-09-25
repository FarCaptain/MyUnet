import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
import random

import os

from TrainingPlateform import TrainingPlateform


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # CPU上tf优化

class LabDialog(QDialog):
    detect_success_signal = pyqtSignal()
    logdir = './savedModels'

    def __init__(self, parent=None):
        super(LabDialog, self).__init__(parent)
        self.setUpUI()
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("训练新模型")
        self.trigger = False

    def setUpUI(self):
        self.resize(1000, 600)
        #print(self.fname)
        normCate = ["True", "False"]
        bnCate = ["True", "False"]
        lossCate = ["dice_loss", "focal_loss(0.25,1)","binary_crossentropy","dice&BCE","dice&focal"]


        self.layout = QFormLayout()
        self.setLayout(self.layout)

        # Label控件
        self.titlelabel = QLabel("训 练 参 数")
        self.nameLabel = QLabel("自定义模型名:")
        self.normLabel = QLabel(" 归一化: ")
        self.bnLabel = QLabel(" 批归一化: ")
        self.batchsizeLabel = QLabel(" 批大小: ")
        self.epochLabel = QLabel(" 轮数: ")
        self.lossLabel = QLabel(" 损失函数: ")

        # button控件
        self.trainButton = QPushButton(" Train！")

        # lineEdit控件
        self.nameEdit = QLineEdit()
        self.normComboBox = QComboBox()
        self.normComboBox.addItems(normCate)
        self.bnComboBox = QComboBox()
        self.bnComboBox.addItems(bnCate)

        self.batchsizeEdit = QLineEdit()
        self.batchsizeEdit.setMaxLength(3)

        self.epochEdit = QLineEdit()
        self.epochEdit.setMaxLength(3)

        self.lossComboBox = QComboBox()
        self.lossComboBox.addItems(lossCate)


        # 添加进formlayout
        self.layout.addRow("", self.titlelabel)
        self.layout.addRow(self.nameLabel, self.nameEdit)
        self.layout.addRow(self.normLabel, self.normComboBox)
        self.layout.addRow(self.bnLabel, self.bnComboBox)
        self.layout.addRow(self.batchsizeLabel, self.batchsizeEdit)
        self.layout.addRow(self.epochLabel, self.epochEdit)
        self.layout.addRow(self.lossLabel, self.lossComboBox)
        self.layout.addRow("", self.trainButton)

        # 设置字体
        font = QFont()
        font.setPixelSize(28)
        self.titlelabel.setFont(font)
        font.setPixelSize(23)
        self.nameLabel.setFont(font)
        self.normLabel.setFont(font)
        self.bnLabel.setFont(font)
        self.batchsizeLabel.setFont(font)
        self.epochLabel.setFont(font)
        self.lossLabel.setFont(font)

        self.normComboBox.setFont(font)
        self.bnComboBox.setFont(font)
        self.batchsizeEdit.setFont(font)
        self.epochEdit.setFont(font)
        self.lossComboBox.setFont(font)

        # button设置
        font.setPixelSize(20)
        self.trainButton.setFont(font)
        self.trainButton.setFixedHeight(32)
        self.trainButton.setFixedWidth(140)

        # 设置间距
        self.titlelabel.setMargin(50)
        self.layout.setVerticalSpacing(10)

        self.trainButton.clicked.connect(self.loading)


    def loading(self):
        self.trainButton.deleteLater()
        self.train = QLabel("训练中...")
        self.layout.addRow("", self.train)
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(78, 100, 450, 25)

        self.timer = QBasicTimer()
        self.step = 0
        self.layout.addRow("", self.pbar)
        self.doAction()

    def startTraining(self):
        T1 = TrainingPlateform(self)

        #set para
        nm = self.nameEdit.text()
        norm = (self.normComboBox.currentText()=="True")
        bn = (self.bnComboBox.currentText()=="True")
        bsize = int(self.batchsizeEdit.text())
        ep = int(self.epochEdit.text())
        tmp = self.lossComboBox.currentText()
        if tmp=="dice_loss":loss=0
        elif tmp=="focal_loss(0.25,1)":loss=1
        elif tmp=="binary_crossentropy":loss=2
        elif tmp=="dice&BCE":loss=3
        else: loss=4

        T1.training(nm,norm,bn,bsize,ep,loss)
        self.rush()
        self.train.setText('训练完成！')
        self.showLossBtn = QPushButton("损失曲线")
        self.showAccBtn = QPushButton("准确率曲线")

        # button设置
        font = QFont()
        font.setPixelSize(20)
        self.showLossBtn.setFont(font)
        self.showLossBtn.setFixedHeight(32)
        self.showLossBtn.setFixedWidth(200)

        self.showAccBtn.setFont(font)
        self.showAccBtn.setFixedHeight(32)
        self.showAccBtn.setFixedWidth(140)

        self.layout.addRow(self.showLossBtn, self.showAccBtn)

        #T1.showLoss()
        self.showLossBtn.clicked.connect(T1.showLoss)
        self.showAccBtn.clicked.connect(T1.showAcc)


    def timerEvent(self, e):
        if self.step >= 100:
            self.timer.stop()
            return
        if self.step == 98:
            self.timer.stop()
            self.startTraining()
            return
        self.step = self.step+2
        self.pbar.setValue(self.step)

    def rush(self):
        self.step = 100
        self.pbar.setValue(self.step)
        return

    def doAction(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(100, self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = LabDialog()
    mainMindow.show()
    sys.exit(app.exec_())
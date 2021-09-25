import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QBasicTimer
import random
import time

class WaitDialog(QWidget):

    def __init__(self):
        super(WaitDialog, self).__init__()

        self.initUI()

    def initUI(self):

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(78, 100, 450, 25)

        #self.btn = QPushButton('开始训练', self)
        #self.btn.move(40, 80)
        #self.btn.clicked.connect(self.doAction)

        self.timer = QBasicTimer()
        self.step = 0

        self.setGeometry(900, 500, 564, 230)
        #self.resize(800,600)
        self.setWindowTitle('训练中，请勿关闭窗口...')
        self.show()
        self.pause = random.randint(0,200)
        self.ind = 0
        self.arr = [20,28,30,42,55,78,90,92,93,94,95,96,97,98,101]
        self.doAction()

        #self.doAction()

    def rush(self):
        if self.step >= 100:
            return
        if not self.timer.isActive():
            self.timer.start()
        return

    def timerEvent(self, e):

        if self.step >= 100:
            self.timer.stop()
            #self.btn.setText('完成')
            self.setWindowTitle('完成！')
            return
        if self.step >= 99:
            self.timer.stop()
            return
        if self.step == self.arr[self.ind]:
            if self.pause==0:
                self.step = self.step + 1
                self.pause = random.randint(0, self.arr[self.ind]+20)
                self.ind+=1
                self.arr[self.ind]
            else:
                self.pause-=1
        else:
            self.step = self.step+1
        self.pbar.setValue(self.step)

    def doAction(self):

        if self.timer.isActive():
            self.timer.stop()
            #self.btn.setText('继续训练')
        elif self.step >= 100:
             self.destroy(self)
        else:
            self.timer.start(100, self)
            #self.btn.setText('暂停训练')

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = WaitDialog()
    ex.show()
    sys.exit(app.exec_())

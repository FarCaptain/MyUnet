import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import *
import qdarkstyle
from SignIn import SignInWidget
#from SignUp import SignUpWidget
import sip
from AdminHome import AdminHome
from WorkerHome import WorkerHome
from ExpertHome import ExpertHome
from changePasswordDialog import changePasswordDialog


class Main(QMainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.layout = QHBoxLayout()
        self.widget = SignInWidget()
        self.resize(900, 600)
        self.setWindowTitle("Weldetect-β")
        self.setCentralWidget(self.widget)
        bar = self.menuBar()
        self.Menu = bar.addMenu("菜单栏")
        #self.signUpAction = QAction("注册", self)
        self.changePasswordAction =QAction("修改密码",self)
        self.signInAction = QAction("登录", self)
        self.quitSignInAction = QAction("退出登录", self)
        self.quitAction = QAction("退出", self)
        #self.Menu.addAction(self.signUpAction)
        self.Menu.addAction(self.changePasswordAction)
        self.Menu.addAction(self.signInAction)
        self.Menu.addAction(self.quitSignInAction)
        self.Menu.addAction(self.quitAction)
        #self.signUpAction.setEnabled(True)
        self.changePasswordAction.setEnabled(True)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(False)
        #admin - expert - worker
        self.widget.is_admin_signal.connect(self.adminSignIn)
        self.widget.is_expert_signal.connect(self.expertSignIn)
        self.widget.is_worker_signal.connect(self.workerSignIn)
        self.Menu.triggered[QAction].connect(self.menuTriggered)

    def adminSignIn(self):
        sip.delete(self.widget)
        self.widget = AdminHome()
        self.setCentralWidget(self.widget)
        self.changePasswordAction.setEnabled(False)
        #self.signUpAction.setEnabled(True)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(True)

    def expertSignIn(self):
        sip.delete(self.widget)
        self.widget = ExpertHome()
        self.setCentralWidget(self.widget)
        self.changePasswordAction.setEnabled(False)
        #self.signUpAction.setEnabled(True)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(True)

    def workerSignIn(self):
        sip.delete(self.widget)
        self.widget = WorkerHome()
        self.setCentralWidget(self.widget)
        self.changePasswordAction.setEnabled(False)
        #self.signUpAction.setEnabled(True)
        self.signInAction.setEnabled(False)
        self.quitSignInAction.setEnabled(True)

    def menuTriggered(self, q):
        if(q.text()=="修改密码"):
            changePsdDialog=changePasswordDialog(self)
            changePsdDialog.show()
            changePsdDialog.exec_()
        if (q.text() == "退出登录"):
            sip.delete(self.widget)
            self.widget = SignInWidget()
            self.setCentralWidget(self.widget)
            self.widget.is_admin_signal.connect(self.adminSignIn)
            self.widget.is_expert_signal.connect(self.expertSignIn)
            self.widget.is_worker_signal.connect(self.workerSignIn)
            #self.signUpAction.setEnabled(True)
            self.changePasswordAction.setEnabled(True)
            self.signInAction.setEnabled(False)
            self.quitSignInAction.setEnabled(False)
        if (q.text() == "登录"):
            sip.delete(self.widget)
            self.widget = SignInWidget()
            self.setCentralWidget(self.widget)
            self.widget.is_admin_signal.connect(self.adminSignIn)
            self.widget.is_expert_signal.connect(self.expertSignIn)
            self.widget.is_worker_signal.connect(self.workerSignIn)
            #self.signUpAction.setEnabled(True)
            self.changePasswordAction.setEnabled(True)
            self.signInAction.setEnabled(False)
            self.quitSignInAction.setEnabled(False)
        if (q.text() == "退出"):
            qApp = QApplication.instance()
            qApp.quit()
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = Main()
    mainMindow.show()
    sys.exit(app.exec_())
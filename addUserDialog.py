import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
import hashlib
from PyQt5.QtSql import *


class addUserDialog(QDialog):
    add_user_success_signal = pyqtSignal()

    def __init__(self, parent=None):
        super(addUserDialog, self).__init__(parent)
        self.setUpUI()
        self.setWindowModality(Qt.WindowModal)
        self.setWindowTitle("添加用户")

    def setUpUI(self):
        UserCategory = ["普通用户", "专家用户"]
        self.resize(300, 400)
        self.layout = QFormLayout()
        self.setLayout(self.layout)

        # Label控件
        self.titlelabel = QLabel("  添加用户")
        self.userIdLabel = QLabel("账    号:")
        self.userNameLabel = QLabel("姓    名:")
        self.categoryLabel = QLabel("用户类别:")
        self.passwordLabel = QLabel("初始密码:")

        # button控件
        self.addUserButton = QPushButton("添 加")

        # lineEdit控件
        self.userIdEdit = QLineEdit()
        self.userNameEdit = QLineEdit()
        self.categoryComboBox = QComboBox()
        self.categoryComboBox.addItems(UserCategory)
        self.passwordEdit = QLineEdit()
        self.passwordEdit.setEchoMode(QLineEdit.Password)

        self.userIdEdit.setMaxLength(20)
        self.userNameEdit.setMaxLength(20)
        self.passwordEdit.setMaxLength(32)

        ## 账号 跟 密码 都只能是字母数字下划线组合 //名字的安全性差一点
        #控制密码格式，有效防止注入攻击
        reg = QRegExp("[a-zA-z0-9]+$")
        pValidator = QRegExpValidator(self)
        pValidator.setRegExp(reg)
        self.passwordEdit.setValidator(pValidator)
        self.userIdEdit.setValidator(pValidator)

        # 添加进formlayout
        self.layout.addRow("", self.titlelabel)
        self.layout.addRow(self.userIdLabel, self.userIdEdit)
        self.layout.addRow(self.userNameLabel, self.userNameEdit)
        self.layout.addRow(self.categoryLabel, self.categoryComboBox)
        self.layout.addRow(self.passwordLabel, self.passwordEdit)
        self.layout.addRow("", self.addUserButton)

        # 设置字体
        font = QFont()
        font.setPixelSize(20)
        self.titlelabel.setFont(font)
        font.setPixelSize(14)
        self.userIdLabel.setFont(font)
        self.userNameLabel.setFont(font)
        self.categoryLabel.setFont(font)
        self.passwordLabel.setFont(font)

        self.userIdEdit.setFont(font)
        self.userNameEdit.setFont(font)
        self.passwordEdit.setFont(font)
        self.categoryComboBox.setFont(font)

        # button设置
        font.setPixelSize(16)
        self.addUserButton.setFont(font)
        self.addUserButton.setFixedHeight(32)
        self.addUserButton.setFixedWidth(140)

        # 设置间距
        self.titlelabel.setMargin(8)
        self.layout.setVerticalSpacing(10)

        self.addUserButton.clicked.connect(self.addUserButtonCicked)

    def addUserButtonCicked(self):
        userId = self.userIdEdit.text()
        name = self.userNameEdit.text()
        tmp = self.categoryComboBox.currentText()
        if(tmp=="专家用户"):
            authority = 1
        else: authority = 0

        password = self.passwordEdit.text()


        if (userId == "" or name == "" or tmp == "" or password == ""):
            print(QMessageBox.warning(self, "警告", "有字段为空，添加失败", QMessageBox.Yes, QMessageBox.Yes))
            return
        else:
            # 加密密码
            hl = hashlib.md5()
            hl.update(password.encode(encoding='utf-8'))
            password = hl.hexdigest()

            db = QSqlDatabase.addDatabase("QSQLITE")
            db.setDatabaseName('./db/demo.db')
            db.open()
            query = QSqlQuery()

            sql = "SELECT * FROM User WHERE UserId='%s'" % (userId)
            query.exec_(sql)
            if (query.next()):
                print(QMessageBox.warning(self, "警告", "该用户已存在，添加失败", QMessageBox.Yes, QMessageBox.Yes))
                return
            else:
                sql = "INSERT INTO User VALUES ('%s','%s','%s','%d')" % (userId, name, password, authority)
            query.exec_(sql)
            db.commit()

            print(QMessageBox.information(self, "提示", "添加用户成功!", QMessageBox.Yes, QMessageBox.Yes))
            self.close()
            self.add_user_success_signal.emit()
            self.clearEdit()
        return

    def clearEdit(self):
        self.userNameEdit.clear()
        self.userIdEdit.clear()
        self.passwordEdit.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = addUserDialog()
    mainMindow.show()
    sys.exit(app.exec_())
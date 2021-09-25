import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import qdarkstyle
from PyQt5.QtSql import *
import sip

class UserManage(QDialog):
    def __init__(self,parent=None):
        super(UserManage, self).__init__(parent)
        self.resize(280, 400)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle("管理用户")
        # 用户数
        self.userCount = 0
        self.oldDeleteId = ""
        self.oldDeleteName = ""
        self.oldDeleteAuthority = ""
        self.deleteId = ""
        self.deleteName = ""
        self.deleteAuthority = ""
        self.setUpUI()

    def setUpUI(self):
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName('./db/demo.db')
        self.db.open()
        self.query = QSqlQuery()
        self.getResult()

        # 表格设置
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(self.userCount)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['账号', '姓名','权限'])
        # 不可编辑
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 标题可拉伸
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 整行选中
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.layout.addWidget(self.tableWidget)
        self.setRows()
        self.deleteUserButton = QPushButton("删 除 用 户")
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.deleteUserButton, Qt.AlignHCenter)
        self.widget = QWidget()
        self.widget.setLayout(hlayout)
        self.widget.setFixedHeight(48)
        font = QFont()
        font.setPixelSize(15)
        self.deleteUserButton.setFixedHeight(36)
        self.deleteUserButton.setFixedWidth(180)
        self.deleteUserButton.setFont(font)
        self.layout.addWidget(self.widget, Qt.AlignCenter)
        # 设置信号
        self.deleteUserButton.clicked.connect(self.deleteUser)
        self.tableWidget.itemClicked.connect(self.getUserInfo)

    def getResult(self): # Sphi
        sql = "SELECT UserId,Name,Authority FROM User WHERE Authority!=2"
        self.query.exec_(sql)
        self.userCount = 0;
        while (self.query.next()):
            self.userCount += 1; # ??
        sql = "SELECT UserId,Name,Authority FROM User WHERE Authority!=2"
        self.query.exec_(sql)

    def setRows(self):
        font = QFont()
        font.setPixelSize(14)
        for i in range(self.userCount):
            if (self.query.next()):
                UserIdItem = QTableWidgetItem(self.query.value(0))
                UserNameItem = QTableWidgetItem(self.query.value(1))
                UserAuthorityItem = QTableWidgetItem(str(self.query.value(2)))

                UserIdItem.setFont(font)
                UserNameItem.setFont(font)
                UserAuthorityItem.setFont(font)
                UserIdItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                UserNameItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                UserAuthorityItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget.setItem(i, 0, UserIdItem)
                self.tableWidget.setItem(i, 1, UserNameItem)
                self.tableWidget.setItem(i, 2, UserAuthorityItem)
        return

    def getUserInfo(self):
        row = self.tableWidget.currentIndex().row()
        self.tableWidget.verticalScrollBar().setSliderPosition(row)
        self.getResult()
        i = 0
        while (self.query.next() and i != row):
            i = i + 1
        self.oldDeleteId = self.deleteId
        self.oldDeleteName = self.deleteName
        self.oldDeleteAuthority = self.deleteAuthority
        self.deleteId = self.query.value(0)
        self.deleteName = self.query.value(1)
        self.deleteAuthority = self.query.value(2)

    def deleteUser(self):
        if (self.deleteId == "" and self.deleteName == ""):
            print(QMessageBox.warning(self, "警告", "请选中要删除的用户", QMessageBox.Yes, QMessageBox.Yes))
            return
        elif (self.deleteId == self.oldDeleteId and self.deleteName == self.oldDeleteName):
            print(QMessageBox.warning(self, "警告", "请选中要删除的用户", QMessageBox.Yes, QMessageBox.Yes))
            return
        if (QMessageBox.information(self, "提醒", "删除用户:%s,%s,%d\n用户一经删除将无法恢复，是否继续?" % (self.deleteId, self.deleteName, self.deleteAuthority),
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No) == QMessageBox.No):
            return
        # 从User表删除用户
        sql = "DELETE FROM User WHERE UserId='%s'" % (self.deleteId)
        self.query.exec_(sql)
        self.db.commit()
        print(QMessageBox.information(self,"提醒","删除用户成功!",QMessageBox.Yes,QMessageBox.Yes))
        self.updateUI()
        return

    def updateUI(self):
        self.getResult()
        self.layout.removeWidget(self.widget)
        self.layout.removeWidget(self.tableWidget)
        sip.delete(self.widget)
        sip.delete(self.tableWidget)
        # 表格设置
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(self.userCount)
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(['账号', '姓名', '权限'])
        # 不可编辑
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # 标题可拉伸
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # 整行选中
        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.layout.addWidget(self.tableWidget)
        self.setRows()
        self.deleteUserButton = QPushButton("删 除 用 户")
        hlayout = QHBoxLayout()
        hlayout.addWidget(self.deleteUserButton, Qt.AlignHCenter)
        self.widget = QWidget()
        self.widget.setLayout(hlayout)
        self.widget.setFixedHeight(48)
        font = QFont()
        font.setPixelSize(15)
        self.deleteUserButton.setFixedHeight(36)
        self.deleteUserButton.setFixedWidth(180)
        self.deleteUserButton.setFont(font)
        self.layout.addWidget(self.widget, Qt.AlignCenter)
        # 设置信号
        self.deleteUserButton.clicked.connect(self.deleteUser)
        self.tableWidget.itemClicked.connect(self.getUserInfo)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = UserManage()
    mainMindow.show()
    sys.exit(app.exec_())
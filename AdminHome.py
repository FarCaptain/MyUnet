# -*- coding: utf-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import qdarkstyle
from PyQt5.QtSql import *
import sqlite3
from addUserDialog import addUserDialog



class AdminHome(QWidget):
    delrow = -1
    conn = sqlite3.connect('./db/demo.db')
    #print("ok!")

    def __init__(self):
        super(AdminHome, self).__init__()
        self.resize(700, 500)
        self.setWindowTitle("欢迎，管理员")
        self.delrow = -1
        # 查询模型
        self.queryModel = None
        # 数据表
        self.tableView = None
        # 当前页
        self.currentPage = 0
        # 总页数
        self.totalPage = 0
        # 总记录数
        self.totalRecord = 0
        # 每页数据数
        self.pageRecord = 10
        self.setUpUI()

    def __del__(self):
        self.conn.close()

    def setUpUI(self):
        self.layout = QVBoxLayout()
        self.Hlayout1 = QHBoxLayout()
        self.Hlayout2 = QHBoxLayout()

        # Hlayout1控件的初始化
        self.searchEdit = QLineEdit()
        self.searchEdit.setFixedHeight(32)
        font = QFont()
        font.setPixelSize(15)
        self.searchEdit.setFont(font)

        self.searchButton = QPushButton("查询")
        self.searchButton.setFixedHeight(32)
        self.searchButton.setFont(font)
        self.searchButton.setIcon(QIcon(QPixmap("./images/search.png")))

        self.condisionComboBox = QComboBox()
        searchCondision = ['按账号查询', '按姓名查询', '按权限查询']
        self.condisionComboBox.setFixedHeight(32)
        self.condisionComboBox.setFont(font)
        self.condisionComboBox.addItems(searchCondision)

        self.Hlayout1.addWidget(self.searchEdit)
        self.Hlayout1.addWidget(self.searchButton)
        self.Hlayout1.addWidget(self.condisionComboBox)

        # Hlayout2初始化
        self.jumpToLabel = QLabel("跳转到第")
        self.pageEdit = QLineEdit()
        self.pageEdit.setFixedWidth(30)
        s = "/" + str(self.totalPage) + "页"
        self.pageLabel = QLabel(s)
        self.jumpToButton = QPushButton("跳转")
        self.prevButton = QPushButton("前一页")
        self.prevButton.setFixedWidth(60)
        self.backButton = QPushButton("后一页")
        self.backButton.setFixedWidth(60)

        Hlayout = QHBoxLayout()
        Hlayout.addWidget(self.jumpToLabel)
        Hlayout.addWidget(self.pageEdit)
        Hlayout.addWidget(self.pageLabel)
        Hlayout.addWidget(self.jumpToButton)
        Hlayout.addWidget(self.prevButton)
        Hlayout.addWidget(self.backButton)
        widget = QWidget()
        widget.setLayout(Hlayout)
        widget.setFixedWidth(470)
        self.Hlayout2.addWidget(widget)

        # tableView
        # open database
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName('./db/demo.db')
        self.db.open()
        self.tableView = QTableView()
        self.tableView.horizontalHeader().setStretchLastSection(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)# 选整行

        self.queryModel = QSqlQueryModel(self)
        self.searchButtonClicked()
        self.tableView.setModel(self.queryModel)

        self.queryModel.setHeaderData(0, Qt.Horizontal, "账号")
        self.queryModel.setHeaderData(1, Qt.Horizontal, "姓名")
        self.queryModel.setHeaderData(2, Qt.Horizontal, "权限")



        self.layout.addLayout(self.Hlayout1)
        self.layout.addWidget(self.tableView)
        self.layout.addLayout(self.Hlayout2)
        self.setLayout(self.layout)


        buttonlayout = QHBoxLayout()

        font = QFont()
        font.setPixelSize(14)
        self.addUserButton = QPushButton("添加用户")
        self.deleteUserButton = QPushButton("删除用户")

        self.addUserButton.setFont(font)
        self.deleteUserButton.setFont(font)
        self.addUserButton.setFixedWidth(180)
        self.addUserButton.setFixedHeight(25)
        self.deleteUserButton.setFixedWidth(180)
        self.deleteUserButton.setFixedHeight(25)



        buttonlayout.addWidget(self.addUserButton, Qt.AlignCenter)
        buttonlayout.addWidget(self.deleteUserButton, Qt.AlignCenter)

        self.layout.addLayout(buttonlayout)


        self.addUserButton.clicked.connect(self.addUserButtonClicked)
        self.searchButton.clicked.connect(self.searchButtonClicked)
        self.prevButton.clicked.connect(self.prevButtonClicked)
        self.backButton.clicked.connect(self.backButtonClicked)
        self.jumpToButton.clicked.connect(self.jumpToButtonClicked)
        self.searchEdit.returnPressed.connect(self.searchButtonClicked)
        self.tableView.clicked.connect(self.findRow)
        self.deleteUserButton.clicked.connect(self.delBtnClicked)

    def addUserButtonClicked(self):
        addDialog = addUserDialog(self)
        addDialog.show()
        addDialog.exec_()
        self.searchButtonClicked() # 刷新页面

    def findRow(self):
        self.delrow = self.tableView.currentIndex().row();
        print("here", self.delrow)
        return

    def setButtonStatus(self):
        if(self.currentPage==self.totalPage):
            self.prevButton.setEnabled(True)
            self.backButton.setEnabled(False)
        if(self.currentPage==1):
            self.backButton.setEnabled(True)
            self.prevButton.setEnabled(False)
        if(self.currentPage<self.totalPage and self.currentPage>1):
            self.prevButton.setEnabled(True)
            self.backButton.setEnabled(True)

    # 得到记录数
    def getTotalRecordCount(self):
        self.queryModel.setQuery("SELECT * FROM User WHERE Authority!=2")
        self.totalRecord = self.queryModel.rowCount()
        print(self.totalRecord)
        return

    # 得到总页数
    def getPageCount(self):
        self.getTotalRecordCount()
        # 上取整
        self.totalPage = int((self.totalRecord + self.pageRecord - 1) / self.pageRecord)
        return

    # 分页记录查询
    def recordQuery(self, index):
        queryCondition = ""
        conditionChoice = self.condisionComboBox.currentText()
        if (conditionChoice == "按账号查询"):
            conditionChoice = 'UserId'
        elif (conditionChoice == "按姓名查询"):
            conditionChoice = 'Name'
        else:
            conditionChoice = 'Authority'

        if (self.searchEdit.text() == ""):
            queryCondition = "SELECT UserId,Name,Authority FROM User WHERE Authority!=2"
            self.queryModel.setQuery(queryCondition)
            self.totalRecord = self.queryModel.rowCount()
            self.totalPage = int((self.totalRecord + self.pageRecord - 1) / self.pageRecord)
            label = "/" + str(int(self.totalPage)) + "页"
            self.pageLabel.setText(label)
            queryCondition = ("select UserId,Name,Authority from User WHERE Authority!=2 ORDER BY %s limit %d,%d" % (conditionChoice,index, self.pageRecord))
            self.queryModel.setQuery(queryCondition)
            self.setButtonStatus()
            return

        # 得到模糊查询条件
        temp = self.searchEdit.text()
        s = '%'
        for i in range(0, len(temp)):
            s = s + temp[i] + "%"
        queryCondition = ("SELECT UserId,Name,Authority FROM User WHERE Authority!=2 AND %s LIKE '%s' ORDER BY %s" % (conditionChoice, s,conditionChoice))
        self.queryModel.setQuery(queryCondition)
        self.totalRecord = self.queryModel.rowCount()
        # 当查询无记录时的操作
        if(self.totalRecord==0):
            print(QMessageBox.information(self,"提醒","查询无记录",QMessageBox.Yes,QMessageBox.Yes))
            queryCondition = "SELECT UserId,Name,Authority FROM User WHERE Authority!=2"
            self.queryModel.setQuery(queryCondition)
            self.totalRecord = self.queryModel.rowCount()
            self.totalPage = int((self.totalRecord + self.pageRecord - 1) / self.pageRecord)
            label = "/" + str(int(self.totalPage)) + "页"
            self.pageLabel.setText(label)
            queryCondition = ("select UserId,Name,Authority from User WHERE Authority!=2 ORDER BY %s limit %d,%d" % (conditionChoice,index, self.pageRecord))
            self.queryModel.setQuery(queryCondition)
            self.setButtonStatus()
            return
        # 总要先得到页码
        self.totalPage = int((self.totalRecord + self.pageRecord - 1) / self.pageRecord)
        label = "/" + str(int(self.totalPage)) + "页"
        self.pageLabel.setText(label)
        queryCondition = ("SELECT UserId,Name,Authority FROM User WHERE Authority!=2 AND %s LIKE '%s' ORDER BY %s LIMIT %d,%d" % (
            conditionChoice, s, conditionChoice,index, self.pageRecord))
        self.queryModel.setQuery(queryCondition)
        self.setButtonStatus()
        return

    # 点击查询
    def searchButtonClicked(self):
        self.currentPage = 1
        self.pageEdit.setText(str(self.currentPage))
        self.getPageCount()
        s = "/" + str(int(self.totalPage)) + "页"
        self.pageLabel.setText(s)
        index = (self.currentPage - 1) * self.pageRecord
        self.recordQuery(index)
        return

    # 向前翻页
    def prevButtonClicked(self):
        self.currentPage -= 1
        if (self.currentPage <= 1):
            self.currentPage = 1
        self.pageEdit.setText(str(self.currentPage))
        index = (self.currentPage - 1) * self.pageRecord
        self.recordQuery(index)
        return

    # 向后翻页
    def backButtonClicked(self):
        self.currentPage += 1
        if (self.currentPage >= int(self.totalPage)):
            self.currentPage = int(self.totalPage)
        self.pageEdit.setText(str(self.currentPage))
        index = (self.currentPage - 1) * self.pageRecord
        self.recordQuery(index)
        return

    # 点击跳转
    def jumpToButtonClicked(self):
        if (self.pageEdit.text().isdigit()):
            self.currentPage = int(self.pageEdit.text())
            if (self.currentPage > self.totalPage):
                self.currentPage = self.totalPage
            if (self.currentPage <= 1):
                self.currentPage = 1
        else:
            self.currentPage = 1
        index = (self.currentPage - 1) * self.pageRecord
        self.pageEdit.setText(str(self.currentPage))
        self.recordQuery(index)
        return

    # 删除用户
    def delBtnClicked(self):
        if self.delrow == -1:
            return
        # get UserId
        index = self.queryModel.index(self.delrow,0)# pmode类型为QSqlQueryModel * 0为指定字段
        delid = self.queryModel.data(index)
        #self.conn.open()
        c = self.conn.cursor()
        c.execute("DELETE from User where UserId = '%s' ;"%delid)
        self.conn.commit()
        #setSql and updateUI
        print(delid)
        self.searchButtonClicked()
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("./images/MainWindow_1.png"))
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    mainMindow = AdminHome()
    mainMindow.show()
    sys.exit(app.exec_())
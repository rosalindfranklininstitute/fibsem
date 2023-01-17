# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'connect.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(932, 850)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setStyleSheet("background-color rgb(255, 255, 255)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 441, 541))
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")
        self.Connection = QtWidgets.QWidget()
        self.Connection.setObjectName("Connection")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.Connection)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(110, 10, 160, 80))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.ConnectButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.ConnectButton.setObjectName("ConnectButton")
        self.verticalLayout.addWidget(self.ConnectButton)
        self.DisconnectButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.DisconnectButton.setObjectName("DisconnectButton")
        self.verticalLayout.addWidget(self.DisconnectButton)
        self.tabWidget.addTab(self.Connection, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 421, 455))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.autocontrast_enable = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.autocontrast_enable.setObjectName("autocontrast_enable")
        self.gridLayout_2.addWidget(self.autocontrast_enable, 3, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 5, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.label_13 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 3, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.ResetImage = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.ResetImage.setObjectName("ResetImage")
        self.gridLayout_2.addWidget(self.ResetImage, 15, 2, 1, 2)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 13, 0, 1, 4)
        self.res_width = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        self.res_width.setMaximum(10000)
        self.res_width.setObjectName("res_width")
        self.gridLayout_2.addWidget(self.res_width, 1, 1, 1, 1)
        self.res_height = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        self.res_height.setMaximum(10000)
        self.res_height.setObjectName("res_height")
        self.gridLayout_2.addWidget(self.res_height, 1, 3, 1, 1)
        self.gamma_enabled = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.gamma_enabled.setObjectName("gamma_enabled")
        self.gridLayout_2.addWidget(self.gamma_enabled, 4, 1, 1, 1)
        self.reset_image_settings = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.reset_image_settings.setObjectName("reset_image_settings")
        self.gridLayout_2.addWidget(self.reset_image_settings, 15, 0, 1, 2)
        self.save_button = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.save_button.setObjectName("save_button")
        self.gridLayout_2.addWidget(self.save_button, 12, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_18.setObjectName("label_18")
        self.gridLayout_2.addWidget(self.label_18, 8, 2, 1, 1)
        self.check_EB = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.check_EB.setFont(font)
        self.check_EB.setObjectName("check_EB")
        self.gridLayout_2.addWidget(self.check_EB, 12, 3, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 0, 0, 1, 4, QtCore.Qt.AlignLeft)
        self.dwell_time_setting = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget_2)
        self.dwell_time_setting.setObjectName("dwell_time_setting")
        self.gridLayout_2.addWidget(self.dwell_time_setting, 2, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 8, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.autosave_enable = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        self.autosave_enable.setObjectName("autosave_enable")
        self.gridLayout_2.addWidget(self.autosave_enable, 7, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 9, 0, 1, 4)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 14, 0, 1, 4)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 2, 2, 1, 1)
        self.check_IB = QtWidgets.QCheckBox(self.gridLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.check_IB.setFont(font)
        self.check_IB.setObjectName("check_IB")
        self.gridLayout_2.addWidget(self.check_IB, 12, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 4, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 2, 1, 1)
        self.open_filepath = QtWidgets.QToolButton(self.gridLayoutWidget_2)
        self.open_filepath.setObjectName("open_filepath")
        self.gridLayout_2.addWidget(self.open_filepath, 5, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout_2.addWidget(self.label_19, 7, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.take_image = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.take_image.setObjectName("take_image")
        self.gridLayout_2.addWidget(self.take_image, 12, 0, 1, 1)
        self.hfw_box = QtWidgets.QSpinBox(self.gridLayoutWidget_2)
        self.hfw_box.setMaximum(5000)
        self.hfw_box.setObjectName("hfw_box")
        self.gridLayout_2.addWidget(self.hfw_box, 8, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 11, 0, 1, 4)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 2, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.savepath_text = QtWidgets.QLabel(self.gridLayoutWidget_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.savepath_text.setFont(font)
        self.savepath_text.setAutoFillBackground(True)
        self.savepath_text.setStyleSheet("savepath_text->setStyleSheet(\"Qlabel { background-color: red)\")")
        self.savepath_text.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.savepath_text.setFrameShadow(QtWidgets.QFrame.Raised)
        self.savepath_text.setText("")
        self.savepath_text.setObjectName("savepath_text")
        self.gridLayout_2.addWidget(self.savepath_text, 5, 2, 1, 2)
        self.RefImage = QtWidgets.QPushButton(self.gridLayoutWidget_2)
        self.RefImage.setObjectName("RefImage")
        self.gridLayout_2.addWidget(self.RefImage, 10, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 6, 0, 1, 1)
        self.image_label = QtWidgets.QTextEdit(self.gridLayoutWidget_2)
        self.image_label.setObjectName("image_label")
        self.gridLayout_2.addWidget(self.image_label, 6, 1, 1, 3)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(20, 10, 401, 261))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_39 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_39.setObjectName("label_39")
        self.gridLayout.addWidget(self.label_39, 3, 6, 1, 1)
        self.move_abs_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.move_abs_button.setObjectName("move_abs_button")
        self.gridLayout.addWidget(self.move_abs_button, 8, 2, 1, 1)
        self.dYchange = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.dYchange.setDecimals(4)
        self.dYchange.setMinimum(-100.0)
        self.dYchange.setObjectName("dYchange")
        self.gridLayout.addWidget(self.dYchange, 3, 5, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_28.setAlignment(QtCore.Qt.AlignCenter)
        self.label_28.setObjectName("label_28")
        self.gridLayout.addWidget(self.label_28, 4, 4, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_38.setObjectName("label_38")
        self.gridLayout.addWidget(self.label_38, 5, 6, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_25.setFont(font)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.gridLayout.addWidget(self.label_25, 1, 1, 1, 3)
        self.label_29 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_29.setAlignment(QtCore.Qt.AlignCenter)
        self.label_29.setObjectName("label_29")
        self.gridLayout.addWidget(self.label_29, 2, 4, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_41.setObjectName("label_41")
        self.gridLayout.addWidget(self.label_41, 4, 6, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_37.setObjectName("label_37")
        self.gridLayout.addWidget(self.label_37, 6, 3, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.gridLayout.addWidget(self.label_22, 4, 1, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_36.setObjectName("label_36")
        self.gridLayout.addWidget(self.label_36, 5, 3, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_34.setObjectName("label_34")
        self.gridLayout.addWidget(self.label_34, 3, 3, 1, 1)
        self.dTchange = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.dTchange.setMaximum(360.0)
        self.dTchange.setObjectName("dTchange")
        self.gridLayout.addWidget(self.dTchange, 6, 5, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_32.setFont(font)
        self.label_32.setAlignment(QtCore.Qt.AlignCenter)
        self.label_32.setObjectName("label_32")
        self.gridLayout.addWidget(self.label_32, 1, 4, 1, 3)
        self.label_26 = QtWidgets.QLabel(self.gridLayoutWidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_26.setFont(font)
        self.label_26.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.gridLayout.addWidget(self.label_26, 0, 1, 1, 6)
        self.move_rel_button = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.move_rel_button.setObjectName("move_rel_button")
        self.gridLayout.addWidget(self.move_rel_button, 8, 5, 1, 1)
        self.zAbs = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.zAbs.setDecimals(4)
        self.zAbs.setMinimum(-100.0)
        self.zAbs.setObjectName("zAbs")
        self.gridLayout.addWidget(self.zAbs, 4, 2, 1, 1)
        self.tAbs = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.tAbs.setMaximum(360.0)
        self.tAbs.setObjectName("tAbs")
        self.gridLayout.addWidget(self.tAbs, 6, 2, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.gridLayout.addWidget(self.label_21, 3, 1, 1, 1)
        self.label_42 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_42.setObjectName("label_42")
        self.gridLayout.addWidget(self.label_42, 2, 6, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.gridLayout.addWidget(self.label_20, 2, 1, 1, 1)
        self.dXchange = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.dXchange.setDecimals(4)
        self.dXchange.setMinimum(-100.0)
        self.dXchange.setObjectName("dXchange")
        self.gridLayout.addWidget(self.dXchange, 2, 5, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_31.setAlignment(QtCore.Qt.AlignCenter)
        self.label_31.setObjectName("label_31")
        self.gridLayout.addWidget(self.label_31, 5, 4, 1, 1)
        self.dRchange = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.dRchange.setMaximum(360.0)
        self.dRchange.setObjectName("dRchange")
        self.gridLayout.addWidget(self.dRchange, 5, 5, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_35.setObjectName("label_35")
        self.gridLayout.addWidget(self.label_35, 4, 3, 1, 1)
        self.rAbs = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.rAbs.setMaximum(360.0)
        self.rAbs.setObjectName("rAbs")
        self.gridLayout.addWidget(self.rAbs, 5, 2, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_27.setAlignment(QtCore.Qt.AlignCenter)
        self.label_27.setObjectName("label_27")
        self.gridLayout.addWidget(self.label_27, 6, 4, 1, 1)
        self.yAbs = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.yAbs.setDecimals(4)
        self.yAbs.setMinimum(-100.0)
        self.yAbs.setObjectName("yAbs")
        self.gridLayout.addWidget(self.yAbs, 3, 2, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.gridLayout.addWidget(self.label_23, 5, 1, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_33.setObjectName("label_33")
        self.gridLayout.addWidget(self.label_33, 2, 3, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.gridLayout.addWidget(self.label_24, 6, 1, 1, 1)
        self.xAbs = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.xAbs.setDecimals(4)
        self.xAbs.setMinimum(-100.0)
        self.xAbs.setObjectName("xAbs")
        self.gridLayout.addWidget(self.xAbs, 2, 2, 1, 1)
        self.dZchange = QtWidgets.QDoubleSpinBox(self.gridLayoutWidget)
        self.dZchange.setDecimals(4)
        self.dZchange.setMinimum(-100.0)
        self.dZchange.setObjectName("dZchange")
        self.gridLayout.addWidget(self.dZchange, 4, 5, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 6, 6, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_30.setAlignment(QtCore.Qt.AlignCenter)
        self.label_30.setObjectName("label_30")
        self.gridLayout.addWidget(self.label_30, 3, 4, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem4, 7, 3, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 540, 441, 271))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.microscope_status = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.microscope_status.setText("")
        self.microscope_status.setAlignment(QtCore.Qt.AlignCenter)
        self.microscope_status.setObjectName("microscope_status")
        self.verticalLayout_2.addWidget(self.microscope_status)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.CLog = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog.setText("")
        self.CLog.setObjectName("CLog")
        self.verticalLayout_2.addWidget(self.CLog)
        self.CLog2 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog2.setText("")
        self.CLog2.setObjectName("CLog2")
        self.verticalLayout_2.addWidget(self.CLog2)
        self.CLog3 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog3.setText("")
        self.CLog3.setObjectName("CLog3")
        self.verticalLayout_2.addWidget(self.CLog3)
        self.CLog4 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog4.setText("")
        self.CLog4.setObjectName("CLog4")
        self.verticalLayout_2.addWidget(self.CLog4)
        self.CLog5 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog5.setText("")
        self.CLog5.setObjectName("CLog5")
        self.verticalLayout_2.addWidget(self.CLog5)
        self.CLog6 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog6.setText("")
        self.CLog6.setObjectName("CLog6")
        self.verticalLayout_2.addWidget(self.CLog6)
        self.CLog7 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog7.setText("")
        self.CLog7.setObjectName("CLog7")
        self.verticalLayout_2.addWidget(self.CLog7)
        self.CLog8 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.CLog8.setText("")
        self.CLog8.setObjectName("CLog8")
        self.verticalLayout_2.addWidget(self.CLog8)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 932, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ConnectButton.setText(_translate("MainWindow", "Connect to Microscope"))
        self.DisconnectButton.setText(_translate("MainWindow", "Disconnect from Microscope"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Connection), _translate("MainWindow", "General"))
        self.autocontrast_enable.setText(_translate("MainWindow", "Enabled"))
        self.label_14.setText(_translate("MainWindow", "Save Path"))
        self.label_13.setText(_translate("MainWindow", "Autocontrast"))
        self.ResetImage.setText(_translate("MainWindow", "Reset Images"))
        self.gamma_enabled.setText(_translate("MainWindow", "Enabled"))
        self.reset_image_settings.setText(_translate("MainWindow", "Reset To Default Settings"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        self.label_11.setText(_translate("MainWindow", "Resolution (WxH)"))
        self.label_18.setText(_translate("MainWindow", "microns"))
        self.check_EB.setText(_translate("MainWindow", "Electron Beam"))
        self.label_10.setText(_translate("MainWindow", "Image Settings"))
        self.label_17.setText(_translate("MainWindow", "Horizontal Field Width"))
        self.autosave_enable.setText(_translate("MainWindow", "Enabled "))
        self.label_16.setText(_translate("MainWindow", "microseconds"))
        self.check_IB.setText(_translate("MainWindow", "Ion Beam"))
        self.label_15.setText(_translate("MainWindow", "Gamma "))
        self.label_5.setText(_translate("MainWindow", "x"))
        self.open_filepath.setText(_translate("MainWindow", "..."))
        self.label_19.setText(_translate("MainWindow", "Autosave"))
        self.take_image.setText(_translate("MainWindow", "Take Image"))
        self.label_12.setText(_translate("MainWindow", "Dwell time"))
        self.RefImage.setText(_translate("MainWindow", "Take Reference Images"))
        self.label.setText(_translate("MainWindow", "Label"))
        self.image_label.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Imaging"))
        self.label_39.setText(_translate("MainWindow", "mm"))
        self.move_abs_button.setText(_translate("MainWindow", "MOVE"))
        self.label_28.setText(_translate("MainWindow", "dZ:"))
        self.label_38.setText(_translate("MainWindow", "degrees"))
        self.label_25.setText(_translate("MainWindow", "Absolute Position"))
        self.label_29.setText(_translate("MainWindow", "dX:"))
        self.label_41.setText(_translate("MainWindow", "mm"))
        self.label_37.setText(_translate("MainWindow", "degrees"))
        self.label_22.setText(_translate("MainWindow", "Z:"))
        self.label_36.setText(_translate("MainWindow", "degrees"))
        self.label_34.setText(_translate("MainWindow", "mm"))
        self.label_32.setText(_translate("MainWindow", "Relative Move"))
        self.label_26.setText(_translate("MainWindow", "Stage Movement"))
        self.move_rel_button.setText(_translate("MainWindow", "MOVE"))
        self.label_21.setText(_translate("MainWindow", "Y:"))
        self.label_42.setText(_translate("MainWindow", "mm"))
        self.label_20.setText(_translate("MainWindow", "X:"))
        self.label_31.setText(_translate("MainWindow", "dR:"))
        self.label_35.setText(_translate("MainWindow", "mm"))
        self.label_27.setText(_translate("MainWindow", "dT:"))
        self.label_23.setText(_translate("MainWindow", "Rotation:"))
        self.label_33.setText(_translate("MainWindow", "mm"))
        self.label_24.setText(_translate("MainWindow", "Tilt:"))
        self.label_40.setText(_translate("MainWindow", "degrees"))
        self.label_30.setText(_translate("MainWindow", "dY:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Movement"))
        self.label_3.setText(_translate("MainWindow", "Console Log"))
